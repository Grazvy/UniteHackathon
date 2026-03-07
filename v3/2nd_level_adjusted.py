"""
Fundamental cold-start upgrade: two-stage approach (UPDATED: item = eclass + manufacturer)

Stage 1 (Candidates): learn buyer–item affinity with a ranking model that uses ONLY
buyer metadata + item popularity/meta features (no pair-history features).
This works for cold buyers.

Stage 2 (Value): for candidates, estimate
ExpectedSavings = P(buy) * E(spend|buy) * SAVINGS_RATE
Core decision: ExpectedSavings > FEE

Warm buyers:
- Candidates = their historical (buyer,item) pairs (same as before, but item = eclass|manufacturer)
- Value models = your original pair-history models (warm_clf, warm_reg)

Cold buyers:
- Candidates = top-K from ranker scores (cold_ranker) over item universe
- Value models = cold_clf, cold_reg trained with cold-available features + ranker score

Rolling validation:
- tune only on net € directly
- track warm and cold-proxy separately (pseudo-cold buyers)

Output: ../submission.csv

NOTE:
- This code assumes customer_test.csv contains both warm and cold buyer metadata
- plis_training.csv contains historical transactions with:
  orderdate, legal_entity_id, eclass, manufacturer, quantityvalue, vk_per_item
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SAVINGS_RATE = 0.15
FEE = 15.0
TOTAL_FEE = FEE

FEATURE_CUTOFF = pd.Timestamp("2024-12-31")
LABEL_END = pd.Timestamp("2025-06-30")

ROLLING_FOLDS = [
    {
        "name": "2024H2",
        "train_end": pd.Timestamp("2024-06-30"),
        "val_start": pd.Timestamp("2024-07-01"),
        "val_end": pd.Timestamp("2024-12-31"),
    },
    {
        "name": "2025H1",
        "train_end": pd.Timestamp("2024-12-31"),
        "val_start": pd.Timestamp("2025-01-01"),
        "val_end": pd.Timestamp("2025-06-30"),
    },
]

# Candidate settings
CANDIDATE_K_COLD = 800          # how many candidates per cold buyer from ranker
NEG_PER_POS = 40                # negatives per positive when fitting ranker / cold models

# Small tuning grid: affects decision thresholding & candidate depth
EXPECTED_SPEND_SCALE_GRID = [1.0, 1.1, 1.2, 1.3]
CANDIDATE_K_COLD_GRID = [400, 800, 1200]

# LightGBM params
LGBM_PARAMS_WARM_CLF = dict(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=25,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

LGBM_PARAMS_WARM_REG = dict(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=25,
    random_state=42,
    verbose=-1,
)

# Ranker for candidates (pairwise learning-to-rank)
LGBM_PARAMS_RANKER = dict(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Cold models: only cold-available features (+ rank_score)
LGBM_PARAMS_COLD_CLF = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

LGBM_PARAMS_COLD_REG = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    random_state=42,
    verbose=-1,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "unite"
OUT = Path(__file__).resolve().parent.parent
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers: warm pair-history features (UPDATED: group by item_key)
# -----------------------------------------------------------------------------
def build_pair_features(df: pd.DataFrame, cutoff: pd.Timestamp, buyer_info: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (buyer, item_key) with pair-history aggregates."""
    grp = df.groupby(["legal_entity_id", "item_key"])
    feats = pd.concat(
        [
            grp["line_value"].sum().rename("total_spend"),
            grp.size().rename("n_orders"),
            grp["ym"].nunique().rename("n_months_active"),
            grp["orderdate"].max().rename("last_order"),
            grp["orderdate"].min().rename("first_order"),
        ],
        axis=1,
    ).reset_index()

    feats["recency_months"] = (cutoff - feats["last_order"]).dt.days / 30
    span_months = ((feats["last_order"] - feats["first_order"]).dt.days / 30).clip(lower=1)
    feats["span_months"] = span_months
    feats["monthly_avg"] = feats["total_spend"] / span_months

    last6 = (
        df[df["orderdate"] > cutoff - pd.DateOffset(months=6)]
        .groupby(["legal_entity_id", "item_key"])["line_value"]
        .sum()
        .rename("last6m_spend")
        .reset_index()
    )
    feats = feats.merge(last6, on=["legal_entity_id", "item_key"], how="left")
    feats["last6m_spend"] = feats["last6m_spend"].fillna(0)

    buyer_total = (
        df.groupby("legal_entity_id")["line_value"].sum().rename("buyer_total_spend").reset_index()
    )
    item_total = (
        df.groupby("item_key")["line_value"].sum().rename("item_total_spend").reset_index()
    )
    feats = feats.merge(buyer_total, on="legal_entity_id", how="left")
    feats = feats.merge(item_total, on="item_key", how="left")

    feats = feats.merge(
        buyer_info[
            ["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]
        ],
        on="legal_entity_id",
        how="left",
    )
    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        feats[c] = feats[c].fillna(-1)

    return feats.drop(columns=["last_order", "first_order"])


def infer_expected_spend_warmpair(
    feats: pd.DataFrame,
    clf,
    reg,
    feature_cols: list[str],
    expected_spend_scale: float,
) -> pd.DataFrame:
    """Warm model inference on pair-history features."""
    X = feats[feature_cols].fillna(0)
    p_buy = clf.predict_proba(X)[:, 1]
    spend_if_buy = np.expm1(reg.predict(X)).clip(min=0)

    out = feats.copy()
    out["p_buy"] = p_buy
    out["spend_if_buy"] = spend_if_buy
    out["expected_spend"] = p_buy * spend_if_buy * expected_spend_scale
    out["expected_savings"] = out["expected_spend"] * SAVINGS_RATE
    out["is_core"] = (out["expected_savings"] > TOTAL_FEE).astype(int)
    return out


def train_two_models_warmpair(train_feats: pd.DataFrame, feature_cols: list[str]):
    """Train warm classifier+regressor on pair-history features."""
    X = train_feats[feature_cols].fillna(0)
    clf = lgb.LGBMClassifier(**LGBM_PARAMS_WARM_CLF)
    clf.fit(X, train_feats["buy_label"])

    pos_mask = train_feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS_WARM_REG)
    reg.fit(X.loc[pos_mask], train_feats.loc[pos_mask, "log_spend_label"])
    return clf, reg


def save_feature_importance_plots(clf, reg, feature_cols: list[str], out_dir: Path) -> None:
    """Feature importance for warm models."""
    clf_imp = pd.Series(clf.feature_importances_, index=feature_cols, name="clf_importance")
    reg_imp = pd.Series(reg.feature_importances_, index=feature_cols, name="reg_importance")
    imp_df = pd.concat([clf_imp, reg_imp], axis=1).fillna(0)
    imp_df["clf_norm"] = imp_df["clf_importance"] / max(imp_df["clf_importance"].sum(), 1)
    imp_df["reg_norm"] = imp_df["reg_importance"] / max(imp_df["reg_importance"].sum(), 1)
    imp_df["combined_importance"] = 0.5 * imp_df["clf_norm"] + 0.5 * imp_df["reg_norm"]

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    imp_df.sort_values("clf_importance", ascending=True)["clf_importance"].plot.barh(
        ax=axes[0], color="#3b82f6", edgecolor="white"
    )
    axes[0].set_title("Warm Classifier Importance\nP(purchase)")
    axes[0].set_xlabel("Importance")

    imp_df.sort_values("reg_importance", ascending=True)["reg_importance"].plot.barh(
        ax=axes[1], color="#10b981", edgecolor="white"
    )
    axes[1].set_title("Warm Regressor Importance\nSpend if purchased")
    axes[1].set_xlabel("Importance")

    imp_df.sort_values("combined_importance", ascending=True)["combined_importance"].plot.barh(
        ax=axes[2], color="#f59e0b", edgecolor="white"
    )
    axes[2].set_title("Combined Predictive Importance")
    axes[2].set_xlabel("Normalized importance")

    plt.suptitle("Warm Models: Main Predictor Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_file = out_dir / "warm_main_predictor_features.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved feature visualization: {out_file}")


def realized_net(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, int, int, int]:
    """Compute realized net score against known future spend (keys: buyer,item_key)."""
    merged = pred_df[["legal_entity_id", "item_key", "is_core"]].merge(
        truth_df, on=["legal_entity_id", "item_key"], how="left"
    )
    merged["future_spend"] = merged["future_spend"].fillna(0)
    earned = (merged["is_core"] * merged["future_spend"] * SAVINGS_RATE).sum()
    fees = merged["is_core"].sum() * TOTAL_FEE
    net = float(earned - fees)

    profitable = (merged["future_spend"] * SAVINGS_RATE > TOTAL_FEE).astype(int)
    tp = int(((merged["is_core"] == 1) & (profitable == 1)).sum())
    fp = int(((merged["is_core"] == 1) & (profitable == 0)).sum())
    return net, int(merged["is_core"].sum()), tp, fp


# -----------------------------------------------------------------------------
# Cold-start candidate ranker and cold value models (UPDATED: item_key)
# -----------------------------------------------------------------------------
def build_item_meta(plis_hist: pd.DataFrame) -> pd.DataFrame:
    """Item meta features available for all buyers; unique per item_key=(eclass, manufacturer)."""
    e = plis_hist.groupby("item_key", as_index=False).agg(
        item_total_spend=("line_value", "sum"),
        item_n_orders=("line_value", "size"),
        item_n_buyers=("legal_entity_id", "nunique"),
    )
    e["log_item_total_spend"] = np.log1p(e["item_total_spend"])
    e["log_item_n_orders"] = np.log1p(e["item_n_orders"])
    e["log_item_n_buyers"] = np.log1p(e["item_n_buyers"])
    e = e.drop_duplicates(subset=["item_key"]).reset_index(drop=True)
    return e


def sample_negative_pairs(
    buyers: np.ndarray,
    positives_by_buyer: dict,
    all_items: np.ndarray,
    n_neg_per_pos: int,
    rng: np.random.RandomState,
    max_group_size: int = 9000,
    max_pos_per_buyer: int | None = 3000,
) -> pd.DataFrame:
    """
    Uniform negative sampling for (buyer,item_key) with per-buyer cap to satisfy
    LightGBM ranker constraint: <= 10000 rows per query(group).
    """
    rows = []
    all_set = set(all_items.tolist())

    for b in buyers:
        pos = positives_by_buyer.get(b, set())
        if not pos:
            continue

        pos_list = list(pos)

        # Optional: cap positives used per buyer (keeps group size manageable)
        if max_pos_per_buyer is not None and len(pos_list) > max_pos_per_buyer:
            pos_list = rng.choice(pos_list, size=max_pos_per_buyer, replace=False).tolist()

        n_pos = len(pos_list)
        if n_pos == 0:
            continue

        # Compute allowed negatives so that total group size <= max_group_size
        max_neg_allowed = max_group_size - n_pos
        if max_neg_allowed <= 0:
            # still emit positives (or you can skip this buyer)
            for it in pos_list:
                rows.append((b, it, 1))
            continue

        target_neg = min(n_pos * n_neg_per_pos, len(all_set) - len(pos), max_neg_allowed)
        if target_neg <= 0:
            for it in pos_list:
                rows.append((b, it, 1))
            continue

        # Oversample then filter out positives
        cand = rng.choice(all_items, size=target_neg * 3, replace=True).tolist()
        negs = []
        for it in cand:
            if it not in pos:
                negs.append(it)
                if len(negs) >= target_neg:
                    break

        # If still short (rare), keep as-is
        for it in pos_list:
            rows.append((b, it, 1))
        for it in negs:
            rows.append((b, it, 0))

    return pd.DataFrame(rows, columns=["legal_entity_id", "item_key", "label"])



def make_cold_rank_features(
    pairs: pd.DataFrame,
    buyer_info: pd.DataFrame,
    item_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Features for ranker/cold models: buyer meta + item meta (+ categorical item_key)."""
    df = pairs.merge(
        buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
        on="legal_entity_id",
        how="left",
    ).merge(
        item_meta[["item_key", "log_item_total_spend", "log_item_n_orders", "log_item_n_buyers"]],
        on="item_key",
        how="left",
    )
    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        df[c] = df[c].fillna(-1)
    for c in ["log_item_total_spend", "log_item_n_orders", "log_item_n_buyers"]:
        df[c] = df[c].fillna(0)

    df["item_key_cat"] = df["item_key"].astype("category")

    if df.duplicated(subset=["legal_entity_id", "item_key"]).any():
        df = df.drop_duplicates(subset=["legal_entity_id", "item_key"]).reset_index(drop=True)

    return df


COLD_FEATS_NUM = [
    "log_employees",
    "section_enc",
    "nace_2digits",
    "has_secondary_nace",
    "log_item_total_spend",
    "log_item_n_orders",
    "log_item_n_buyers",
]
COLD_FEATS_CAT = ["item_key_cat"]


def train_ranker(
    plis_hist: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set,
    train_end: pd.Timestamp,
    rng: np.random.RandomState,
    neg_per_pos: int = NEG_PER_POS,
):
    """Train LGBMRanker for candidate generation over item_key."""
    hist = plis_hist[(plis_hist["orderdate"] <= train_end) & (plis_hist["legal_entity_id"].isin(warm_train_ids))].copy()
    if hist.empty:
        return None, None

    item_meta = build_item_meta(hist)

    pos_pairs = hist.groupby(["legal_entity_id", "item_key"]).size().reset_index(name="cnt")
    positives_by_buyer = {
        b: set(g["item_key"].tolist())
        for b, g in pos_pairs.groupby("legal_entity_id")
    }

    all_items = np.array(sorted(hist["item_key"].unique().tolist()))
    buyers = np.array(sorted(list(positives_by_buyer.keys())))

    MAX_GROUP_SIZE_RANKER = 9000  # must be < 10000 (LightGBM limit). Keep buffer.
    MAX_POS_PER_BUYER_RANKER = 3000  # optional; prevents huge buyers from dominating

    sampled = sample_negative_pairs(
        buyers=buyers,
        positives_by_buyer=positives_by_buyer,
        all_items=all_items,
        n_neg_per_pos=neg_per_pos,
        rng=rng,
        max_group_size=MAX_GROUP_SIZE_RANKER,
        max_pos_per_buyer=MAX_POS_PER_BUYER_RANKER,
    )

    feats = make_cold_rank_features(sampled, buyer_info, item_meta)

    X = feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
    y = feats["label"].astype(int)

    group_sizes = feats.groupby("legal_entity_id").size().loc[buyers].tolist()

    ranker = lgb.LGBMRanker(**LGBM_PARAMS_RANKER)
    ranker.fit(
        X, y,
        group=group_sizes,
        categorical_feature=COLD_FEATS_CAT,
    )
    return ranker, item_meta


def score_candidates_for_buyers(
    buyer_ids: list,
    all_items: np.ndarray,
    buyer_info: pd.DataFrame,
    item_meta: pd.DataFrame,
    ranker,
    top_k: int,
) -> pd.DataFrame:
    """Score (buyer,item_key) pairs for a buyer set and keep top_k per buyer."""
    rows = []
    for b in buyer_ids:
        tmp = pd.DataFrame({"legal_entity_id": b, "item_key": all_items})
        tmp["label"] = 0
        tmp_feats = make_cold_rank_features(tmp, buyer_info, item_meta)
        X = tmp_feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
        tmp_feats["rank_score"] = ranker.predict(X)
        tmp_feats = tmp_feats.sort_values("rank_score", ascending=False).head(top_k)
        rows.append(tmp_feats[["legal_entity_id", "item_key", "rank_score"]])
    if not rows:
        return pd.DataFrame(columns=["legal_entity_id", "item_key", "rank_score"])
    return pd.concat(rows, ignore_index=True)


def build_future_spend_labels(
    plis: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ids: set,
) -> pd.DataFrame:
    """Future spend by (buyer,item_key) in label window."""
    lab = plis[
        (plis["orderdate"] >= start) & (plis["orderdate"] <= end) & (plis["legal_entity_id"].isin(ids))
    ].copy()
    out = (
        lab.groupby(["legal_entity_id", "item_key"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    return out


def train_cold_value_models(
    plis_hist: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set,
    fold: dict,
    ranker,
    item_meta: pd.DataFrame,
    rng: np.random.RandomState,
    neg_per_pos: int = NEG_PER_POS,
):
    """
    Train cold P(buy) and spend|buy using cold-available features + rank_score.
    Labels come from fold val window, but features only from metadata + item_meta + rank_score.
    """
    train_end, val_start, val_end = fold["train_end"], fold["val_start"], fold["val_end"]

    hist = plis_hist[(plis_hist["orderdate"] <= train_end) & (plis_hist["legal_entity_id"].isin(warm_train_ids))].copy()
    if hist.empty:
        return None, None

    all_items = np.array(sorted(hist["item_key"].unique().tolist()))

    truth = build_future_spend_labels(plis_hist, val_start, val_end, warm_train_ids)
    positives_by_buyer = {
        b: set(g.loc[g["future_spend"] > 0, "item_key"].tolist())
        for b, g in truth.groupby("legal_entity_id")
    }
    buyers = np.array(sorted(list(positives_by_buyer.keys())))
    if len(buyers) == 0:
        return None, None

    sampled = sample_negative_pairs(
        buyers=buyers,
        positives_by_buyer=positives_by_buyer,
        all_items=all_items,
        n_neg_per_pos=neg_per_pos,
        rng=rng,
    )

    sampled = sampled.merge(truth, on=["legal_entity_id", "item_key"], how="left")
    sampled["future_spend"] = sampled["future_spend"].fillna(0.0)

    feats = make_cold_rank_features(sampled, buyer_info, item_meta)
    X_rank = feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
    feats["rank_score"] = ranker.predict(X_rank)
    feats["log_spend_label"] = np.log1p(feats["future_spend"])

    COLD_VALUE_FEATS = COLD_FEATS_NUM + ["rank_score"] + COLD_FEATS_CAT

    clf = lgb.LGBMClassifier(**LGBM_PARAMS_COLD_CLF)
    clf.fit(
        feats[COLD_VALUE_FEATS],
        (feats["future_spend"] > 0).astype(int),
        categorical_feature=COLD_FEATS_CAT,
    )

    pos_mask = feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS_COLD_REG)
    if pos_mask.sum() > 20:
        reg.fit(
            feats.loc[pos_mask, COLD_VALUE_FEATS],
            feats.loc[pos_mask, "log_spend_label"],
            categorical_feature=COLD_FEATS_CAT,
        )
    else:
        reg.fit(
            feats[COLD_VALUE_FEATS].head(50),
            np.zeros(min(50, len(feats))),
            categorical_feature=COLD_FEATS_CAT,
        )
    return clf, reg


def infer_expected_spend_cold(
    candidate_pairs: pd.DataFrame,   # legal_entity_id,item_key,rank_score(optional)
    buyer_info: pd.DataFrame,
    item_meta: pd.DataFrame,
    ranker,
    cold_clf,
    cold_reg,
    expected_spend_scale: float,
) -> pd.DataFrame:
    """Cold inference using buyer meta + item meta + rank_score."""

    base = candidate_pairs.copy()
    if "rank_score" not in base.columns:
        base["rank_score"] = np.nan

    feats = make_cold_rank_features(base.assign(label=0), buyer_info, item_meta)

    feats = feats.merge(
        base[["legal_entity_id", "item_key", "rank_score"]],
        on=["legal_entity_id", "item_key"],
        how="left",
        suffixes=("", "_base"),
    )
    needs_score = feats["rank_score"].isna()
    if needs_score.any():
        Xr = feats.loc[needs_score, COLD_FEATS_NUM + COLD_FEATS_CAT]
        feats.loc[needs_score, "rank_score"] = ranker.predict(Xr)

    COLD_VALUE_FEATS = COLD_FEATS_NUM + ["rank_score"] + COLD_FEATS_CAT
    X = feats[COLD_VALUE_FEATS]

    p_buy = cold_clf.predict_proba(X)[:, 1]
    spend_if_buy = np.expm1(cold_reg.predict(X)).clip(min=0)

    out = feats[["legal_entity_id", "item_key", "rank_score"]].copy()
    out["p_buy"] = p_buy
    out["spend_if_buy"] = spend_if_buy
    out["expected_spend"] = p_buy * spend_if_buy * expected_spend_scale
    out["expected_savings"] = out["expected_spend"] * SAVINGS_RATE
    out["is_core"] = (out["expected_savings"] > TOTAL_FEE).astype(int)
    return out


# -----------------------------------------------------------------------------
# Rolling evaluation (warm + pseudo-cold via true cold pipeline)
# -----------------------------------------------------------------------------
WARM_FEATURE_COLS = [
    "total_spend",
    "n_orders",
    "n_months_active",
    "recency_months",
    "span_months",
    "monthly_avg",
    "last6m_spend",
    "buyer_total_spend",
    "item_total_spend",
    "log_employees",
    "section_enc",
    "nace_2digits",
    "has_secondary_nace",
]


def evaluate_fold(
    plis: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set,
    warm_eval_ids: set,
    pseudo_cold_ids: set,
    fold: dict,
    expected_spend_scale: float,
    candidate_k_cold: int,
    rng: np.random.RandomState,
) -> dict:
    train_end, val_start, val_end = fold["train_end"], fold["val_start"], fold["val_end"]

    feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_train_ids))].copy()
    label_df = plis[
        (plis["orderdate"] >= val_start) & (plis["orderdate"] <= val_end) &
        (plis["legal_entity_id"].isin(warm_train_ids.union(warm_eval_ids).union(pseudo_cold_ids)))
    ].copy()

    train_feats = build_pair_features(feat_df, train_end, buyer_info)
    future_spend_train = (
        label_df[label_df["legal_entity_id"].isin(warm_train_ids)]
        .groupby(["legal_entity_id", "item_key"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    train_feats = train_feats.merge(future_spend_train, on=["legal_entity_id", "item_key"], how="left")
    train_feats["future_spend"] = train_feats["future_spend"].fillna(0)
    train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
    train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

    if train_feats.empty:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    warm_clf, warm_reg = train_two_models_warmpair(train_feats, WARM_FEATURE_COLS)

    ranker, item_meta = train_ranker(plis, buyer_info, warm_train_ids, train_end, rng=rng, neg_per_pos=NEG_PER_POS)
    if ranker is None:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    cold_clf, cold_reg = train_cold_value_models(
        plis_hist=plis,
        buyer_info=buyer_info,
        warm_train_ids=warm_train_ids,
        fold=fold,
        ranker=ranker,
        item_meta=item_meta,
        rng=rng,
        neg_per_pos=NEG_PER_POS,
    )
    if cold_clf is None:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    warm_eval_feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_eval_ids))].copy()
    warm_eval_feats = build_pair_features(warm_eval_feat_df, train_end, buyer_info)
    warm_eval_pred = infer_expected_spend_warmpair(
        warm_eval_feats, warm_clf, warm_reg, WARM_FEATURE_COLS, expected_spend_scale
    )

    warm_truth = (
        label_df[label_df["legal_entity_id"].isin(warm_eval_ids)]
        .groupby(["legal_entity_id", "item_key"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_eval_pred, warm_truth)

    hist = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_train_ids))].copy()
    all_items = np.array(sorted(hist["item_key"].unique().tolist()))
    pseudo_cold_list = sorted(list(pseudo_cold_ids))

    cand = score_candidates_for_buyers(
        buyer_ids=pseudo_cold_list,
        all_items=all_items,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        top_k=candidate_k_cold,
    )

    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cand,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=expected_spend_scale,
    )

    cold_truth = (
        label_df[label_df["legal_entity_id"].isin(pseudo_cold_ids)]
        .groupby(["legal_entity_id", "item_key"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    cold_net, cold_n, cold_tp, cold_fp = realized_net(cold_pred, cold_truth)

    return {
        "fold": fold["name"],
        "warm_net": warm_net,
        "warm_preds": warm_n,
        "warm_tp": warm_tp,
        "warm_fp": warm_fp,
        "cold_proxy_net": cold_net,
        "cold_proxy_preds": cold_n,
        "cold_proxy_tp": cold_tp,
        "cold_proxy_fp": cold_fp,
        "combined_net": warm_net + cold_net,
    }


# -----------------------------------------------------------------------------
# 1) Load data
# -----------------------------------------------------------------------------
print("Loading data...")
customers = pd.read_csv(DATA / "customer_test.csv", sep="\t")
customers["nace_code"] = customers["nace_code"].astype(int)

nace = pd.read_csv(DATA / "nace_codes.csv", sep="\t")
nace["nace_code"] = nace["nace_code"].astype(int)

sections = sorted(nace["toplevel_section"].dropna().unique())
section_map = {s: i for i, s in enumerate(sections)}

buyer_info = customers.merge(
    nace[["nace_code", "toplevel_section", "nace_2digits"]],
    on="nace_code",
    how="left",
)
buyer_info["section_enc"] = buyer_info["toplevel_section"].map(section_map).fillna(-1).astype(int)
buyer_info["nace_2digits"] = buyer_info["nace_2digits"].fillna(-1)
buyer_info["has_secondary_nace"] = buyer_info["secondary_nace_code"].notna().astype(int)
buyer_info["log_employees"] = np.log1p(buyer_info["estimated_number_employees"].fillna(0))

warm_ids = set(customers.loc[customers["task"] == "predict future", "legal_entity_id"])
cold_ids = set(customers.loc[customers["task"] == "cold start", "legal_entity_id"])

chunks = []
for chunk in pd.read_csv(
    DATA / "plis_training.csv",
    sep="\t",
    low_memory=False,
    usecols=["orderdate", "legal_entity_id", "eclass", "manufacturer", "quantityvalue", "vk_per_item"],
    chunksize=300_000,
):
    chunk["orderdate"] = pd.to_datetime(chunk["orderdate"])
    chunks.append(chunk)

plis = pd.concat(chunks, ignore_index=True)

# eclass cleaning
plis["eclass"] = pd.to_numeric(plis["eclass"], errors="coerce")
plis = plis.dropna(subset=["eclass"])
plis["eclass"] = plis["eclass"].astype(int).astype(str)

# manufacturer cleaning
plis["manufacturer"] = plis["manufacturer"].astype(str).fillna("UNKNOWN")

# build item_key
plis["item_key"] = plis["eclass"] + "|" + plis["manufacturer"]

plis["line_value"] = plis["quantityvalue"] * plis["vk_per_item"]
plis["ym"] = plis["orderdate"].dt.to_period("M")
plis = plis[plis["legal_entity_id"].isin(set(customers["legal_entity_id"]))].copy()

print(f"Rows: {len(plis):,} | Warm buyers: {len(warm_ids)} | Cold buyers: {len(cold_ids)}")


# -----------------------------------------------------------------------------
# 2) Rolling validation + net tuning
# -----------------------------------------------------------------------------
print("\nRunning rolling validation + net-score tuning...")

rng = np.random.RandomState(42)
warm_list = np.array(sorted(list(warm_ids)))
rng.shuffle(warm_list)

pseudo_cold_size = max(8, int(len(warm_list) * 0.2))
pseudo_cold_ids = set(warm_list[:pseudo_cold_size].tolist())

warm_train_ids = warm_ids - pseudo_cold_ids
warm_eval_ids = warm_ids - pseudo_cold_ids

print(
    f"Warm train buyers: {len(warm_train_ids)} | "
    f"Warm eval buyers: {len(warm_eval_ids)} | "
    f"Pseudo-cold buyers: {len(pseudo_cold_ids)}"
)

search_rows = []
for scale, kcold in product(EXPECTED_SPEND_SCALE_GRID, CANDIDATE_K_COLD_GRID):
    fold_metrics = []
    for fold in ROLLING_FOLDS:
        fold_metrics.append(
            evaluate_fold(
                plis=plis,
                buyer_info=buyer_info,
                warm_train_ids=warm_train_ids,
                warm_eval_ids=warm_eval_ids,
                pseudo_cold_ids=pseudo_cold_ids,
                fold=fold,
                expected_spend_scale=scale,
                candidate_k_cold=kcold,
                rng=np.random.RandomState(123 + int(scale * 100) + kcold),
            )
        )
    fold_df = pd.DataFrame(fold_metrics)
    search_rows.append(
        {
            "scale": scale,
            "candidate_k_cold": kcold,
            "avg_warm_net": float(fold_df["warm_net"].mean()),
            "avg_cold_proxy_net": float(fold_df["cold_proxy_net"].mean()),
            "avg_combined_net": float(fold_df["combined_net"].mean()),
        }
    )

search_df = pd.DataFrame(search_rows).sort_values("avg_combined_net", ascending=False).reset_index(drop=True)
print("\nTop tuning configs (objective = avg_combined_net):")
print(search_df.head(10).to_string(index=False))

best = search_df.iloc[0]
EXPECTED_SPEND_SCALE = float(best["scale"])
CANDIDATE_K_COLD = int(best["candidate_k_cold"])

print("\nChosen params:")
print(f" EXPECTED_SPEND_SCALE = {EXPECTED_SPEND_SCALE}")
print(f" CANDIDATE_K_COLD = {CANDIDATE_K_COLD}")
print(f" Avg warm net = EUR {best['avg_warm_net']:,.2f}")
print(f" Avg cold proxy net = EUR {best['avg_cold_proxy_net']:,.2f}")
print(f" Avg combined net = EUR {best['avg_combined_net']:,.2f}")


# -----------------------------------------------------------------------------
# 3) Final training on canonical split
# -----------------------------------------------------------------------------
print("\nTraining final models on canonical split...")

feat_df = plis[(plis["orderdate"] <= FEATURE_CUTOFF) & (plis["legal_entity_id"].isin(warm_ids))].copy()
label_df = plis[
    (plis["orderdate"] > FEATURE_CUTOFF) & (plis["orderdate"] <= LABEL_END) & (plis["legal_entity_id"].isin(warm_ids))
].copy()

train_feats = build_pair_features(feat_df, FEATURE_CUTOFF, buyer_info)
future_spend = (
    label_df.groupby(["legal_entity_id", "item_key"])["line_value"]
    .sum()
    .rename("future_spend")
    .reset_index()
)
train_feats = train_feats.merge(future_spend, on=["legal_entity_id", "item_key"], how="left")
train_feats["future_spend"] = train_feats["future_spend"].fillna(0)
train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

warm_clf, warm_reg = train_two_models_warmpair(train_feats, WARM_FEATURE_COLS)
save_feature_importance_plots(warm_clf, warm_reg, WARM_FEATURE_COLS, PLOTS)

warm_holdout_pred = infer_expected_spend_warmpair(
    train_feats[["legal_entity_id", "item_key"] + WARM_FEATURE_COLS],
    warm_clf,
    warm_reg,
    WARM_FEATURE_COLS,
    EXPECTED_SPEND_SCALE,
)
warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_holdout_pred, future_spend)
print("\nCanonical warm holdout metrics (approx, optimistic):")
print(f" warm_net = EUR {warm_net:,.2f} | preds={warm_n} | tp={warm_tp} | fp={warm_fp}")

final_rng = np.random.RandomState(42)
ranker, item_meta = train_ranker(plis, buyer_info, warm_ids, FEATURE_CUTOFF, rng=final_rng, neg_per_pos=NEG_PER_POS)

final_fold_like = {
    "train_end": FEATURE_CUTOFF,
    "val_start": FEATURE_CUTOFF + pd.Timedelta(days=1),
    "val_end": LABEL_END,
}
cold_clf, cold_reg = train_cold_value_models(
    plis_hist=plis,
    buyer_info=buyer_info,
    warm_train_ids=warm_ids,
    fold=final_fold_like,
    ranker=ranker,
    item_meta=item_meta,
    rng=np.random.RandomState(7),
    neg_per_pos=NEG_PER_POS,
)


# -----------------------------------------------------------------------------
# 4) Final predictions: warm
# -----------------------------------------------------------------------------
pred_cutoff = plis["orderdate"].max()

warm_full = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
warm_pred_feats = build_pair_features(warm_full, pred_cutoff, buyer_info)
warm_pred = infer_expected_spend_warmpair(warm_pred_feats, warm_clf, warm_reg, WARM_FEATURE_COLS, EXPECTED_SPEND_SCALE)

warm_sub = (
    warm_pred.loc[warm_pred["is_core"] == 1, ["legal_entity_id", "item_key"]]
    .rename(columns={"legal_entity_id": "buyer_id", "item_key": "cluster"})
)
print(f"Warm core predictions: {len(warm_sub):,}")


# -----------------------------------------------------------------------------
# 5) Final predictions: cold
# -----------------------------------------------------------------------------
hist_for_universe = plis[(plis["orderdate"] <= pred_cutoff) & (plis["legal_entity_id"].isin(warm_ids))].copy()
all_items = np.array(sorted(hist_for_universe["item_key"].unique().tolist()))

cold_list = sorted(list(cold_ids))
if len(cold_list) == 0:
    cold_sub = pd.DataFrame(columns=["buyer_id", "cluster"])
    print("Cold: no cold buyers")
else:
    cold_cand = score_candidates_for_buyers(
        buyer_ids=cold_list,
        all_items=all_items,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        top_k=CANDIDATE_K_COLD,
    )
    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cold_cand,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=EXPECTED_SPEND_SCALE,
    )

    cold_sub = (
        cold_pred.loc[cold_pred["is_core"] == 1, ["legal_entity_id", "item_key"]]
        .rename(columns={"legal_entity_id": "buyer_id", "item_key": "cluster"})
    )
    covered = cold_sub["buyer_id"].nunique() if not cold_sub.empty else 0
    print(f"Cold core predictions: {len(cold_sub):,} across {covered}/{len(cold_ids)} buyers")


# -----------------------------------------------------------------------------
# 6) Submission
# -----------------------------------------------------------------------------
submission = (
    pd.concat([warm_sub, cold_sub], ignore_index=True)
    .drop_duplicates()
    .sort_values(["buyer_id", "cluster"])
    .reset_index(drop=True)
)
submission.to_csv(OUT / "submission.csv", index=False)
print(f"Saved submission.csv with {len(submission):,} rows ({len(warm_sub)} warm + {len(cold_sub)} cold)")
