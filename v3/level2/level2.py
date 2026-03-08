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
SAVINGS_RATE = 0.10
FEE = 10.0
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
NEG_PER_POS = 100             # negatives per positive when fitting ranker / cold models

# Fixed submission size: total number of (buyer, item) predictions to submit.
# All candidates are ranked by expected_savings (descending) and the top N are selected

# Small tuning grid: affects decision thresholding & candidate depth
EXPECTED_SPEND_SCALE_GRID = [0.6]
CANDIDATE_K_COLD_GRID = [500]
# Level-1 hierarchical gating:
# Only Level-2 candidates whose parent eclass is in the buyer's top-K L1 eclasses pass.
# Set to None to disable gating (pure feature-only mode).
L1_GATE_TOP_K: int | None = 150  # how many eclasses per buyer pass the L1 gate
L1_RANKER_N_ESTIMATORS = 300   # lighter ranker — same architecture, fewer trees
MAX_MANUFACTURERS_PER_ECLASS: int = 20  # Option B: keep top-K manufacturers per (buyer, eclass)
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

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "unite"
OUT = Path(__file__).resolve().parent.parent.parent
PLOTS = Path(__file__).resolve().parent.parent / "plots"
PLOTS.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers: warm pair-history features (UPDATED: group by item_key)
# -----------------------------------------------------------------------------
def build_pair_features(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    buyer_info: pd.DataFrame,
    l1_scores: dict | None = None,
) -> pd.DataFrame:
    """Build one row per (buyer, item_key) with pair-history aggregates + L1 prior."""
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

    # Level-1 prior: look up eclass rank score for each (buyer, eclass)
    if l1_scores is not None and len(l1_scores) > 0:
        eclass_col = feats["item_key"].str.split("|", n=1).str[0]
        feats["l1_rank_score"] = [
            get_l1_score(l1_scores, int(b), ec)
            for b, ec in zip(feats["legal_entity_id"], eclass_col)
        ]
    else:
        feats["l1_rank_score"] = 0.0

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


def deduplicate_manufacturers(
    pred_df: pd.DataFrame,
    key_col: str = "item_key",
    max_per_eclass: int = MAX_MANUFACTURERS_PER_ECLASS,
) -> pd.DataFrame:
    """For each (buyer, eclass), keep only the top-`max_per_eclass` item_keys by expected_savings."""
    pred_df = pred_df.copy()
    pred_df["_eclass"] = pred_df[key_col].str.split("|", n=1).str[0]
    pred_df = (
        pred_df.sort_values("expected_savings", ascending=False)
               .groupby(["legal_entity_id", "_eclass"], sort=False, group_keys=False)
               .head(max_per_eclass)
               .drop(columns=["_eclass"])
               .reset_index(drop=True)
    )
    return pred_df


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
# Level-1 (eclass) helpers: used as hierarchical prior for Level-2
# -----------------------------------------------------------------------------
def _build_l1_ranker_params() -> dict:
    p = dict(LGBM_PARAMS_RANKER)
    p["n_estimators"] = L1_RANKER_N_ESTIMATORS
    return p


def build_l1_scores(
    plis_hist: pd.DataFrame,
    buyer_info: pd.DataFrame,
    train_end: pd.Timestamp,
    warm_ids: set,
    rng: np.random.RandomState,
    neg_per_pos: int = 20,
) -> dict[tuple[int, str], float]:
    """Train a lightweight Level-1 eclass ranker and return a dict
    {(legal_entity_id, eclass): rank_score} for all warm buyers.

    This score is later used as a feature and gating signal in Level-2.
    """
    hist = plis_hist[(plis_hist["orderdate"] <= train_end) & (plis_hist["legal_entity_id"].isin(warm_ids))].copy()
    if hist.empty:
        return {}

    # Build eclass meta
    eclass_meta = (
        hist.groupby("eclass", as_index=False)
        .agg(
            log_eclass_total_spend=("line_value", lambda x: np.log1p(x.sum())),
            log_eclass_n_orders=("line_value", "size"),
            log_eclass_n_buyers=("legal_entity_id", "nunique"),
        )
    )
    eclass_meta["log_eclass_n_orders"] = np.log1p(eclass_meta["log_eclass_n_orders"])
    eclass_meta["log_eclass_n_buyers"] = np.log1p(eclass_meta["log_eclass_n_buyers"])

    l1_cold_feats_num = [
        "log_employees", "section_enc", "nace_2digits", "has_secondary_nace",
        "log_eclass_total_spend", "log_eclass_n_orders", "log_eclass_n_buyers",
    ]

    pos_pairs = hist.groupby(["legal_entity_id", "eclass"]).size().reset_index(name="cnt")
    positives_by_buyer: dict[int, set] = {}
    for b, g in pos_pairs.groupby("legal_entity_id"):
        positives_by_buyer[int(b)] = set(g["eclass"].tolist())

    all_eclasses = np.array(sorted(hist["eclass"].unique().tolist()))
    buyers = np.array(sorted(positives_by_buyer.keys()))

    # simplified negative sampling (no group-size cap needed for eclass universe)
    rows = []
    all_set = set(all_eclasses.tolist())
    for b in buyers:
        pos = positives_by_buyer.get(int(b), set())
        if not pos:
            continue
        pos_list = list(pos)
        n_neg = min(len(all_set) - len(pos), len(pos_list) * neg_per_pos, 8000)
        cand = rng.choice(all_eclasses, size=n_neg * 3, replace=True).tolist()
        negs = []
        for c in cand:
            if c not in pos:
                negs.append(c)
                if len(negs) >= n_neg:
                    break
        for ec in pos_list:
            rows.append((int(b), ec, 1))
        for ec in negs:
            rows.append((int(b), ec, 0))

    if not rows:
        return {}

    sampled = pd.DataFrame(rows, columns=["legal_entity_id", "eclass", "label"])
    df = sampled.merge(
        buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
        on="legal_entity_id", how="left",
    ).merge(eclass_meta, on="eclass", how="left")
    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        df[c] = df[c].fillna(-1)
    for c in ["log_eclass_total_spend", "log_eclass_n_orders", "log_eclass_n_buyers"]:
        df[c] = df[c].fillna(0)
    df["eclass_cat"] = df["eclass"].astype("category")

    group_sizes = df.groupby("legal_entity_id").size().loc[buyers].tolist()
    X = df[l1_cold_feats_num + ["eclass_cat"]]
    y = df["label"].astype(int)

    l1_ranker = lgb.LGBMRanker(**_build_l1_ranker_params())
    l1_ranker.fit(X, y, group=group_sizes, categorical_feature=["eclass_cat"])

    # Score every (buyer, eclass) pair for all warm buyers
    score_dict: dict[tuple[int, str], float] = {}
    for b in buyers:
        tmp = pd.DataFrame({"legal_entity_id": int(b), "eclass": all_eclasses})
        tmp = tmp.merge(
            buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
            on="legal_entity_id", how="left",
        ).merge(eclass_meta, on="eclass", how="left")
        for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
            tmp[c] = tmp[c].fillna(-1)
        for c in ["log_eclass_total_spend", "log_eclass_n_orders", "log_eclass_n_buyers"]:
            tmp[c] = tmp[c].fillna(0)
        tmp["eclass_cat"] = tmp["eclass"].astype("category")
        scores = l1_ranker.predict(tmp[l1_cold_feats_num + ["eclass_cat"]])
        for ec, sc in zip(all_eclasses, scores):
            score_dict[(int(b), str(ec))] = float(sc)
    return score_dict


def get_l1_score(l1_scores: dict, buyer_id: int, eclass: str) -> float:
    """Look up L1 rank score; returns 0.0 if not found (new buyers / unseen eclass)."""
    return l1_scores.get((int(buyer_id), str(eclass)), 0.0)


def get_l1_top_eclasses(l1_scores: dict, buyer_id: int, top_k: int) -> set[str]:
    """Return the set of top-K eclasses by L1 rank score for a buyer."""
    buyer_scores = [
        (ec, sc) for (b, ec), sc in l1_scores.items() if b == int(buyer_id)
    ]
    if not buyer_scores:
        return set()
    buyer_scores.sort(key=lambda x: x[1], reverse=True)
    return {ec for ec, _ in buyer_scores[:top_k]}


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
    item_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Popularity-based negative sampling (frequency^0.75 smoothing, word2vec-style)
    with per-buyer cap to satisfy LightGBM ranker constraint: <= 10000 rows per query.

    Popular items are sampled more often as negatives, creating harder negatives.
    Falls back to uniform if no weights given.
    """
    rows = []
    all_set = set(all_items.tolist())

    # Normalise weights once outside the buyer loop
    if item_weights is not None:
        sampling_probs = item_weights ** 0.75
        sampling_probs = sampling_probs / sampling_probs.sum()
    else:
        sampling_probs = None

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
            for it in pos_list:
                rows.append((b, it, 1))
            continue

        target_neg = min(n_pos * n_neg_per_pos, len(all_set) - len(pos), max_neg_allowed)
        if target_neg <= 0:
            for it in pos_list:
                rows.append((b, it, 1))
            continue

        # Oversample then filter out positives
        cand = rng.choice(all_items, size=target_neg * 3, replace=True, p=sampling_probs).tolist()
        negs = []
        for it in cand:
            if it not in pos:
                negs.append(it)
                if len(negs) >= target_neg:
                    break

        for it in pos_list:
            rows.append((b, it, 1))
        for it in negs:
            rows.append((b, it, 0))

    return pd.DataFrame(rows, columns=["legal_entity_id", "item_key", "label"])



def make_cold_rank_features(
    pairs: pd.DataFrame,
    buyer_info: pd.DataFrame,
    item_meta: pd.DataFrame,
    l1_scores: dict | None = None,
) -> pd.DataFrame:
    """Features for ranker/cold models: buyer meta + item meta + L1 prior (+ categorical item_key)."""
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

    # Level-1 prior: eclass-level rank score from the L1 ranker
    if l1_scores is not None and len(l1_scores) > 0:
        # Extract parent eclass from item_key "eclass|manufacturer"
        eclass_col = df["item_key"].str.split("|", n=1).str[0]
        df["l1_rank_score"] = [
            get_l1_score(l1_scores, int(b), ec)
            for b, ec in zip(df["legal_entity_id"], eclass_col)
        ]
    else:
        df["l1_rank_score"] = 0.0

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
    "l1_rank_score",   # Level-1 eclass prior
]
COLD_FEATS_CAT = ["item_key_cat"]


def train_ranker(
    plis_hist: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set,
    train_end: pd.Timestamp,
    rng: np.random.RandomState,
    neg_per_pos: int = NEG_PER_POS,
    l1_scores: dict | None = None,
):
    """Train LGBMRanker for candidate generation over item_key (with L1 prior feature)."""
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

    # Popularity weights: order count per item_key (aligned to all_items order)
    item_order_counts = hist.groupby("item_key").size()
    item_weights = np.array(
        [item_order_counts.get(it, 1) for it in all_items], dtype=float
    )

    sampled = sample_negative_pairs(
        buyers=buyers,
        positives_by_buyer=positives_by_buyer,
        all_items=all_items,
        n_neg_per_pos=neg_per_pos,
        rng=rng,
        max_group_size=MAX_GROUP_SIZE_RANKER,
        max_pos_per_buyer=MAX_POS_PER_BUYER_RANKER,
        item_weights=item_weights,
    )

    feats = make_cold_rank_features(sampled, buyer_info, item_meta, l1_scores=l1_scores)

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
    l1_scores: dict | None = None,
    l1_gate_top_k: int | None = L1_GATE_TOP_K,
) -> pd.DataFrame:
    """Score (buyer,item_key) pairs for a buyer set and keep top_k per buyer.

    Level-1 hierarchical gating: if l1_gate_top_k is set, only candidates whose
    parent eclass is in the buyer's top-K L1 eclasses are scored (rest are dropped).
    This drastically reduces false positives without sacrificing recall on true needs.
    """
    rows = []
    for b in buyer_ids:
        items_for_buyer = all_items

        # Hierarchical gate: keep only items in buyer's top-K eclasses (Level-1)
        if l1_scores and l1_gate_top_k is not None:
            top_eclasses = get_l1_top_eclasses(l1_scores, int(b), l1_gate_top_k)
            if top_eclasses:
                items_for_buyer = np.array(
                    [it for it in all_items if str(it).split("|")[0] in top_eclasses]
                )
            if len(items_for_buyer) == 0:
                items_for_buyer = all_items  # fallback: gate produced empty set

        tmp = pd.DataFrame({"legal_entity_id": b, "item_key": items_for_buyer})
        tmp["label"] = 0
        tmp_feats = make_cold_rank_features(tmp, buyer_info, item_meta, l1_scores=l1_scores)
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
    l1_scores: dict | None = None,
):
    """
    Train cold P(buy) and spend|buy using cold-available features + rank_score + L1 prior.
    Labels come from fold val window, but features only from metadata + item_meta + rank_score.
    """
    train_end, val_start, val_end = fold["train_end"], fold["val_start"], fold["val_end"]

    hist = plis_hist[(plis_hist["orderdate"] <= train_end) & (plis_hist["legal_entity_id"].isin(warm_train_ids))].copy()
    if hist.empty:
        return None, None

    all_items = np.array(sorted(hist["item_key"].unique().tolist()))

    # Popularity weights: order count per item_key (aligned to all_items order)
    item_order_counts = hist.groupby("item_key").size()
    item_weights = np.array(
        [item_order_counts.get(it, 1) for it in all_items], dtype=float
    )

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
        item_weights=item_weights,
    )

    sampled = sampled.merge(truth, on=["legal_entity_id", "item_key"], how="left")
    sampled["future_spend"] = sampled["future_spend"].fillna(0.0)

    feats = make_cold_rank_features(sampled, buyer_info, item_meta, l1_scores=l1_scores)
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
    l1_scores: dict | None = None,
) -> pd.DataFrame:
    """Cold inference using buyer meta + item meta + rank_score + L1 prior."""

    base = candidate_pairs.copy()
    if "rank_score" not in base.columns:
        base["rank_score"] = np.nan

    feats = make_cold_rank_features(base.assign(label=0), buyer_info, item_meta, l1_scores=l1_scores)

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
    "l1_rank_score",   # Level-1 eclass prior
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

    # Build Level-1 scores (eclass prior) for this fold's training window
    l1_scores = build_l1_scores(
        plis_hist=plis,
        buyer_info=buyer_info,
        train_end=train_end,
        warm_ids=warm_train_ids,
        rng=np.random.RandomState(rng.randint(0, 10000)),
    )

    feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_train_ids))].copy()
    label_df = plis[
        (plis["orderdate"] >= val_start) & (plis["orderdate"] <= val_end) &
        (plis["legal_entity_id"].isin(warm_train_ids.union(warm_eval_ids).union(pseudo_cold_ids)))
    ].copy()

    train_feats = build_pair_features(feat_df, train_end, buyer_info, l1_scores=l1_scores)
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

    ranker, item_meta = train_ranker(
        plis, buyer_info, warm_train_ids, train_end, rng=rng,
        neg_per_pos=NEG_PER_POS, l1_scores=l1_scores,
    )
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
        l1_scores=l1_scores,
    )
    if cold_clf is None:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    warm_eval_feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_eval_ids))].copy()
    warm_eval_feats = build_pair_features(warm_eval_feat_df, train_end, buyer_info, l1_scores=l1_scores)
    warm_eval_pred = infer_expected_spend_warmpair(
        warm_eval_feats, warm_clf, warm_reg, WARM_FEATURE_COLS, expected_spend_scale
    )
    warm_eval_pred = deduplicate_manufacturers(warm_eval_pred)

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
        l1_scores=l1_scores,
    )

    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cand,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=expected_spend_scale,
        l1_scores=l1_scores,
    )
    cold_pred = deduplicate_manufacturers(cold_pred)

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

future_spend = (
    label_df.groupby(["legal_entity_id", "item_key"])["line_value"]
    .sum()
    .rename("future_spend")
    .reset_index()
)

# Build L1 scores on full canonical training window
print("Building Level-1 eclass prior scores...")
final_l1_rng = np.random.RandomState(99)
final_l1_scores = build_l1_scores(
    plis_hist=plis,
    buyer_info=buyer_info,
    train_end=FEATURE_CUTOFF,
    warm_ids=warm_ids,
    rng=final_l1_rng,
)
print(f"L1 scores computed for {len(final_l1_scores):,} (buyer,eclass) pairs")

train_feats = build_pair_features(feat_df, FEATURE_CUTOFF, buyer_info, l1_scores=final_l1_scores)
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
ranker, item_meta = train_ranker(
    plis, buyer_info, warm_ids, FEATURE_CUTOFF,
    rng=final_rng, neg_per_pos=NEG_PER_POS, l1_scores=final_l1_scores,
)

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
    l1_scores=final_l1_scores,
)


# -----------------------------------------------------------------------------
# 4) Final predictions: warm
# -----------------------------------------------------------------------------
pred_cutoff = plis["orderdate"].max()

warm_full = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
warm_pred_feats = build_pair_features(warm_full, pred_cutoff, buyer_info, l1_scores=final_l1_scores)
warm_pred = infer_expected_spend_warmpair(warm_pred_feats, warm_clf, warm_reg, WARM_FEATURE_COLS, EXPECTED_SPEND_SCALE)
warm_pred = deduplicate_manufacturers(warm_pred)

warm_sub = (
    warm_pred[["legal_entity_id", "item_key", "expected_savings"]]
    .rename(columns={"legal_entity_id": "buyer_id", "item_key": "cluster"})
)
print(f"Warm candidates (all): {len(warm_sub):,}")


# -----------------------------------------------------------------------------
# 5) Final predictions: cold
# -----------------------------------------------------------------------------
hist_for_universe = plis[(plis["orderdate"] <= pred_cutoff) & (plis["legal_entity_id"].isin(warm_ids))].copy()
all_items = np.array(sorted(hist_for_universe["item_key"].unique().tolist()))

cold_list = sorted(list(cold_ids))
if len(cold_list) == 0:
    cold_sub = pd.DataFrame(columns=["buyer_id", "cluster", "expected_savings"])
    print("Cold: no cold buyers")
else:
    cold_cand = score_candidates_for_buyers(
        buyer_ids=cold_list,
        all_items=all_items,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        top_k=CANDIDATE_K_COLD,
        l1_scores=final_l1_scores,
    )
    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cold_cand,
        buyer_info=buyer_info,
        item_meta=item_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=EXPECTED_SPEND_SCALE,
        l1_scores=final_l1_scores,
    )
    cold_pred = deduplicate_manufacturers(cold_pred)

    cold_sub = (
        cold_pred[["legal_entity_id", "item_key", "expected_savings"]]
        .rename(columns={"legal_entity_id": "buyer_id", "item_key": "cluster"})
    )
    covered = cold_sub["buyer_id"].nunique() if not cold_sub.empty else 0
    print(f"Cold candidates (all): {len(cold_sub):,} across {covered}/{len(cold_ids)} buyers")


# -----------------------------------------------------------------------------
# 6) Submission: rank all candidates by expected_savings, keep top TARGET_PREDICTIONS
# -----------------------------------------------------------------------------
all_candidates = (
    pd.concat([warm_sub, cold_sub], ignore_index=True)
    .drop_duplicates(subset=["buyer_id", "cluster"])
    .sort_values("expected_savings", ascending=False)
    .reset_index(drop=True)
)

submission = (
    all_candidates.head(TARGET_PREDICTIONS)[["buyer_id", "cluster"]]
    .sort_values(["buyer_id", "cluster"])
    .reset_index(drop=True)
)
submission.to_csv(OUT / "submission.csv", index=False)
print(
    f"Saved submission.csv with {len(submission):,} rows "
    f"(top {TARGET_PREDICTIONS} from {len(all_candidates):,} total candidates; "
    f"{len(warm_sub):,} warm + {len(cold_sub):,} cold)"
)
