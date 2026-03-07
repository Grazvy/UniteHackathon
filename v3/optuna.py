"""
Two-stage warm/cold pipeline + Optuna tuning on realized net €

What this script adds:
- Optuna integration
- Parameterized training functions (no hidden global params during tuning)
- Fold caching (precompute fold datasets once; trials only train models)
- Objective = avg combined net € across rolling folds

Cold start approach (fundamental change):
- Candidate generation: LGBMRanker using ONLY buyer meta + eclass meta (+ eclass categorical)
- Cold value models: cold_clf + cold_reg using same cold-available features + rank_score

Warm buyers:
- Original pair-history features + warm clf/reg

Outputs:
- ../submission.csv

Notes:
- Uses pseudo-cold proxy set from warm buyers to tune cold performance (like before)
- Keeps tuning space modest; expand once it runs fast/stable
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import optuna

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

# sampling
NEG_PER_POS_RANK = 30
NEG_PER_POS_COLD = 30

# default candidate depth (will be tuned)
DEFAULT_CANDIDATE_K_COLD = 800
DEFAULT_EXPECTED_SPEND_SCALE = 1.15

# data paths
DATA = Path(__file__).resolve().parent.parent / "data" / "unite"
OUT = Path(__file__).resolve().parent.parent
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)

# warm pair-history features
WARM_FEATURE_COLS = [
    "total_spend",
    "n_orders",
    "n_months_active",
    "recency_months",
    "span_months",
    "monthly_avg",
    "last6m_spend",
    "buyer_total_spend",
    "eclass_total_spend",
    "log_employees",
    "section_enc",
    "nace_2digits",
    "has_secondary_nace",
]

# cold feature set (numeric + categorical)
COLD_FEATS_NUM = [
    "log_employees",
    "section_enc",
    "nace_2digits",
    "has_secondary_nace",
    "log_eclass_total_spend",
    "log_eclass_n_orders",
    "log_eclass_n_buyers",
]
COLD_FEATS_CAT = ["eclass_cat"]


# -----------------------------------------------------------------------------
# Default model params (Optuna will overwrite some fields)
# -----------------------------------------------------------------------------
BASE_WARM_CLF_PARAMS = dict(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=25,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)
BASE_WARM_REG_PARAMS = dict(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=25,
    random_state=42,
    verbose=-1,
)

BASE_RANKER_PARAMS = dict(
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

BASE_COLD_CLF_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)
BASE_COLD_REG_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=25,
    random_state=42,
    verbose=-1,
)


# -----------------------------------------------------------------------------
# Helpers: warm pair-history features
# -----------------------------------------------------------------------------
def build_pair_features(df: pd.DataFrame, cutoff: pd.Timestamp, buyer_info: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["legal_entity_id", "eclass"])
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
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("last6m_spend")
        .reset_index()
    )
    feats = feats.merge(last6, on=["legal_entity_id", "eclass"], how="left")
    feats["last6m_spend"] = feats["last6m_spend"].fillna(0)

    buyer_total = df.groupby("legal_entity_id")["line_value"].sum().rename("buyer_total_spend").reset_index()
    eclass_total = df.groupby("eclass")["line_value"].sum().rename("eclass_total_spend").reset_index()
    feats = feats.merge(buyer_total, on="legal_entity_id", how="left")
    feats = feats.merge(eclass_total, on="eclass", how="left")

    feats = feats.merge(
        buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
        on="legal_entity_id",
        how="left",
    )
    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        feats[c] = feats[c].fillna(-1)

    return feats.drop(columns=["last_order", "first_order"])


def train_two_models_warmpair(train_feats: pd.DataFrame, feature_cols: list[str], clf_params: dict, reg_params: dict):
    X = train_feats[feature_cols].fillna(0)

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(X, train_feats["buy_label"])

    pos_mask = train_feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**reg_params)
    if pos_mask.sum() > 20:
        reg.fit(X.loc[pos_mask], train_feats.loc[pos_mask, "log_spend_label"])
    else:
        reg.fit(X.head(50), np.zeros(min(50, len(X))))

    return clf, reg


def infer_expected_spend_warmpair(
    feats: pd.DataFrame,
    clf,
    reg,
    feature_cols: list[str],
    expected_spend_scale: float,
) -> pd.DataFrame:
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


def realized_net(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, int, int, int]:
    merged = pred_df[["legal_entity_id", "eclass", "is_core"]].merge(
        truth_df, on=["legal_entity_id", "eclass"], how="left"
    )
    merged["future_spend"] = merged["future_spend"].fillna(0.0)

    earned = (merged["is_core"] * merged["future_spend"] * SAVINGS_RATE).sum()
    fees = merged["is_core"].sum() * TOTAL_FEE
    net = float(earned - fees)

    profitable = (merged["future_spend"] * SAVINGS_RATE > TOTAL_FEE).astype(int)
    tp = int(((merged["is_core"] == 1) & (profitable == 1)).sum())
    fp = int(((merged["is_core"] == 1) & (profitable == 0)).sum())
    return net, int(merged["is_core"].sum()), tp, fp


# -----------------------------------------------------------------------------
# Helpers: cold rank features + negative sampling
# -----------------------------------------------------------------------------
def build_eclass_meta(plis_hist: pd.DataFrame) -> pd.DataFrame:
    e = plis_hist.groupby("eclass", as_index=False).agg(
        eclass_total_spend=("line_value", "sum"),
        eclass_n_orders=("line_value", "size"),
        eclass_n_buyers=("legal_entity_id", "nunique"),
    )
    e["log_eclass_total_spend"] = np.log1p(e["eclass_total_spend"])
    e["log_eclass_n_orders"] = np.log1p(e["eclass_n_orders"])
    e["log_eclass_n_buyers"] = np.log1p(e["eclass_n_buyers"])
    return e.drop_duplicates(subset=["eclass"]).reset_index(drop=True)


def make_cold_rank_features(
    pairs: pd.DataFrame,
    buyer_info: pd.DataFrame,
    eclass_meta: pd.DataFrame,
) -> pd.DataFrame:
    df = pairs.merge(
        buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
        on="legal_entity_id",
        how="left",
    ).merge(
        eclass_meta[["eclass", "log_eclass_total_spend", "log_eclass_n_orders", "log_eclass_n_buyers"]],
        on="eclass",
        how="left",
    )

    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        df[c] = df[c].fillna(-1)
    for c in ["log_eclass_total_spend", "log_eclass_n_orders", "log_eclass_n_buyers"]:
        df[c] = df[c].fillna(0)

    df["eclass_cat"] = df["eclass"].astype("category")

    # safety: ensure 1 row per pair
    df = df.drop_duplicates(subset=["legal_entity_id", "eclass"]).reset_index(drop=True)
    return df


def sample_negative_pairs(
    buyers: np.ndarray,
    positives_by_buyer: dict,
    all_eclasses: np.ndarray,
    n_neg_per_pos: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows = []
    all_set = set(all_eclasses.tolist())
    for b in buyers:
        pos = positives_by_buyer.get(b, set())
        if not pos:
            continue

        pos_list = list(pos)
        n_pos = len(pos_list)
        n_neg = min(len(all_set) - len(pos), n_pos * n_neg_per_pos)
        if n_neg <= 0:
            continue

        cand = rng.choice(all_eclasses, size=n_neg * 3, replace=True).tolist()
        negs = []
        for ec in cand:
            if ec not in pos:
                negs.append(ec)
                if len(negs) >= n_neg:
                    break

        for ec in pos_list:
            rows.append((b, ec, 1))
        for ec in negs:
            rows.append((b, ec, 0))

    return pd.DataFrame(rows, columns=["legal_entity_id", "eclass", "label"])


# -----------------------------------------------------------------------------
# Training: ranker + cold value models
# -----------------------------------------------------------------------------
def train_ranker_from_cached(rank_train_feats: pd.DataFrame, group_sizes: list[int], ranker_params: dict):
    X = rank_train_feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
    y = rank_train_feats["label"].astype(int)
    ranker = lgb.LGBMRanker(**ranker_params)
    ranker.fit(X, y, group=group_sizes, categorical_feature=COLD_FEATS_CAT)
    return ranker


def train_cold_value_models_from_cached(cold_train_feats: pd.DataFrame, ranker, cold_clf_params: dict, cold_reg_params: dict):
    # add rank_score
    Xr = cold_train_feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
    feats = cold_train_feats.copy()
    feats["rank_score"] = ranker.predict(Xr)

    COLD_VALUE_FEATS = COLD_FEATS_NUM + ["rank_score"] + COLD_FEATS_CAT

    clf = lgb.LGBMClassifier(**cold_clf_params)
    clf.fit(
        feats[COLD_VALUE_FEATS],
        (feats["future_spend"] > 0).astype(int),
        categorical_feature=COLD_FEATS_CAT,
    )

    pos_mask = feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**cold_reg_params)
    if pos_mask.sum() > 20:
        reg.fit(
            feats.loc[pos_mask, COLD_VALUE_FEATS],
            feats.loc[pos_mask, "log_spend_label"],
            categorical_feature=COLD_FEATS_CAT,
        )
    else:
        reg.fit(feats[COLD_VALUE_FEATS].head(50), np.zeros(min(50, len(feats))), categorical_feature=COLD_FEATS_CAT)
    return clf, reg


def score_candidates_for_buyers_batch(
    buyer_ids: list,
    all_eclasses: np.ndarray,
    buyer_info: pd.DataFrame,
    eclass_meta: pd.DataFrame,
    ranker,
    top_k: int,
) -> pd.DataFrame:
    """
    Batch scoring by building a cartesian product buyer_ids x all_eclasses.
    This can be heavy; for large universes use chunking.
    """
    out_rows = []
    all_e = pd.Series(all_eclasses)

    for b in buyer_ids:
        tmp = pd.DataFrame({"legal_entity_id": b, "eclass": all_e.values})
        tmp["label"] = 0
        feats = make_cold_rank_features(tmp, buyer_info, eclass_meta)
        X = feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
        feats["rank_score"] = ranker.predict(X)
        feats = feats.sort_values("rank_score", ascending=False).head(top_k)
        out_rows.append(feats[["legal_entity_id", "eclass", "rank_score"]])

    if not out_rows:
        return pd.DataFrame(columns=["legal_entity_id", "eclass", "rank_score"])
    return pd.concat(out_rows, ignore_index=True)


def infer_expected_spend_cold(
    candidate_pairs: pd.DataFrame,   # legal_entity_id,eclass,rank_score(optional)
    buyer_info: pd.DataFrame,
    eclass_meta: pd.DataFrame,
    ranker,
    cold_clf,
    cold_reg,
    expected_spend_scale: float,
) -> pd.DataFrame:
    base = candidate_pairs.copy()
    if "rank_score" not in base.columns:
        base["rank_score"] = np.nan

    feats = make_cold_rank_features(base.assign(label=0), buyer_info, eclass_meta)

    # attach rank_score by keys (safe)
    feats = feats.merge(
        base[["legal_entity_id", "eclass", "rank_score"]],
        on=["legal_entity_id", "eclass"],
        how="left",
    )
    needs = feats["rank_score"].isna()
    if needs.any():
        Xr = feats.loc[needs, COLD_FEATS_NUM + COLD_FEATS_CAT]
        feats.loc[needs, "rank_score"] = ranker.predict(Xr)

    COLD_VALUE_FEATS = COLD_FEATS_NUM + ["rank_score"] + COLD_FEATS_CAT
    X = feats[COLD_VALUE_FEATS]
    p_buy = cold_clf.predict_proba(X)[:, 1]
    spend_if_buy = np.expm1(cold_reg.predict(X)).clip(min=0)

    out = feats[["legal_entity_id", "eclass", "rank_score"]].copy()
    out["p_buy"] = p_buy
    out["spend_if_buy"] = spend_if_buy
    out["expected_spend"] = p_buy * spend_if_buy * expected_spend_scale
    out["expected_savings"] = out["expected_spend"] * SAVINGS_RATE
    out["is_core"] = (out["expected_savings"] > TOTAL_FEE).astype(int)
    return out


# -----------------------------------------------------------------------------
# Fold cache building
# -----------------------------------------------------------------------------
@dataclass
class FoldCache:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    # warm
    warm_train_feats: pd.DataFrame
    warm_eval_feats: pd.DataFrame
    warm_truth: pd.DataFrame

    # ranker
    rank_train_feats: pd.DataFrame
    rank_group_sizes: list[int]
    eclass_meta: pd.DataFrame
    all_eclasses: np.ndarray

    # cold value
    cold_train_feats: pd.DataFrame

    # pseudo-cold evaluation target
    pseudo_cold_truth: pd.DataFrame


def build_future_spend_labels(plis: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ids: set) -> pd.DataFrame:
    lab = plis[(plis["orderdate"] >= start) & (plis["orderdate"] <= end) & (plis["legal_entity_id"].isin(ids))].copy()
    out = lab.groupby(["legal_entity_id", "eclass"])["line_value"].sum().rename("future_spend").reset_index()
    return out


def prepare_fold_cache(
    plis: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set,
    warm_eval_ids: set,
    pseudo_cold_ids: set,
    fold: dict,
    base_rng: np.random.RandomState,
) -> FoldCache:
    train_end, val_start, val_end = fold["train_end"], fold["val_start"], fold["val_end"]

    # history for warm train
    hist_train = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_train_ids))].copy()
    if hist_train.empty:
        raise ValueError(f"Empty history for fold {fold['name']}")

    # label window data
    label_df = plis[
        (plis["orderdate"] >= val_start) & (plis["orderdate"] <= val_end) &
        (plis["legal_entity_id"].isin(warm_train_ids.union(warm_eval_ids).union(pseudo_cold_ids)))
    ].copy()

    # warm train feats
    warm_train_feats = build_pair_features(hist_train, train_end, buyer_info)
    future_spend_train = (
        label_df[label_df["legal_entity_id"].isin(warm_train_ids)]
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    warm_train_feats = warm_train_feats.merge(future_spend_train, on=["legal_entity_id", "eclass"], how="left")
    warm_train_feats["future_spend"] = warm_train_feats["future_spend"].fillna(0.0)
    warm_train_feats["buy_label"] = (warm_train_feats["future_spend"] > 0).astype(int)
    warm_train_feats["log_spend_label"] = np.log1p(warm_train_feats["future_spend"])

    # warm eval feats
    hist_eval = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_eval_ids))].copy()
    warm_eval_feats = build_pair_features(hist_eval, train_end, buyer_info)

    warm_truth = (
        label_df[label_df["legal_entity_id"].isin(warm_eval_ids)]
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )

    # Ranker meta
    eclass_meta = build_eclass_meta(hist_train)
    all_eclasses = np.array(sorted(eclass_meta["eclass"].unique().tolist()))

    # ranker training pairs based on HISTORY positives (implicit feedback)
    pos_pairs = hist_train.groupby(["legal_entity_id", "eclass"]).size().reset_index(name="cnt")
    positives_by_buyer = {b: set(g["eclass"].tolist()) for b, g in pos_pairs.groupby("legal_entity_id")}
    buyers = np.array(sorted(list(positives_by_buyer.keys())))

    rng_rank = np.random.RandomState(base_rng.randint(0, 1_000_000))
    sampled_rank = sample_negative_pairs(buyers, positives_by_buyer, all_eclasses, NEG_PER_POS_RANK, rng_rank)
    rank_train_feats = make_cold_rank_features(sampled_rank, buyer_info, eclass_meta)

    # group sizes for ranker
    # ensure order matches "buyers"
    # (rank_train_feats has both pos+neg for each buyer)
    rank_group_sizes = rank_train_feats.groupby("legal_entity_id").size().loc[buyers].tolist()

    # cold value training pairs based on FUTURE positives in val window, but features are cold-only
    truth_train = build_future_spend_labels(plis, val_start, val_end, warm_train_ids)
    positives_future = {b: set(g.loc[g["future_spend"] > 0, "eclass"].tolist()) for b, g in truth_train.groupby("legal_entity_id")}
    buyers2 = np.array(sorted(list(positives_future.keys())))

    rng_cold = np.random.RandomState(base_rng.randint(0, 1_000_000))
    sampled_cold = sample_negative_pairs(buyers2, positives_future, all_eclasses, NEG_PER_POS_COLD, rng_cold)
    sampled_cold = sampled_cold.merge(truth_train, on=["legal_entity_id", "eclass"], how="left")
    sampled_cold["future_spend"] = sampled_cold["future_spend"].fillna(0.0)
    cold_train_feats = make_cold_rank_features(sampled_cold, buyer_info, eclass_meta)
    cold_train_feats["log_spend_label"] = np.log1p(cold_train_feats["future_spend"])

    pseudo_cold_truth = (
        label_df[label_df["legal_entity_id"].isin(pseudo_cold_ids)]
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )

    return FoldCache(
        name=fold["name"],
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        warm_train_feats=warm_train_feats,
        warm_eval_feats=warm_eval_feats,
        warm_truth=warm_truth,
        rank_train_feats=rank_train_feats,
        rank_group_sizes=rank_group_sizes,
        eclass_meta=eclass_meta,
        all_eclasses=all_eclasses,
        cold_train_feats=cold_train_feats,
        pseudo_cold_truth=pseudo_cold_truth,
    )


# -----------------------------------------------------------------------------
# Fold evaluation (cached)
# -----------------------------------------------------------------------------
def evaluate_fold_cached(
    cache: FoldCache,
    buyer_info: pd.DataFrame,
    pseudo_cold_ids: set,
    expected_spend_scale: float,
    candidate_k_cold: int,
    warm_clf_params: dict,
    warm_reg_params: dict,
    ranker_params: dict,
    cold_clf_params: dict,
    cold_reg_params: dict,
) -> dict:
    # train warm models
    warm_clf, warm_reg = train_two_models_warmpair(
        cache.warm_train_feats, WARM_FEATURE_COLS, warm_clf_params, warm_reg_params
    )
    warm_eval_pred = infer_expected_spend_warmpair(
        cache.warm_eval_feats, warm_clf, warm_reg, WARM_FEATURE_COLS, expected_spend_scale
    )
    warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_eval_pred, cache.warm_truth)

    # train ranker
    ranker = train_ranker_from_cached(cache.rank_train_feats, cache.rank_group_sizes, ranker_params)

    # train cold value models
    cold_clf, cold_reg = train_cold_value_models_from_cached(
        cache.cold_train_feats, ranker, cold_clf_params, cold_reg_params
    )

    # score pseudo-cold candidates + infer values
    pseudo_cold_list = sorted(list(pseudo_cold_ids))
    cand = score_candidates_for_buyers_batch(
        buyer_ids=pseudo_cold_list,
        all_eclasses=cache.all_eclasses,
        buyer_info=buyer_info,
        eclass_meta=cache.eclass_meta,
        ranker=ranker,
        top_k=candidate_k_cold,
    )
    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cand,
        buyer_info=buyer_info,
        eclass_meta=cache.eclass_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=expected_spend_scale,
    )
    cold_net, cold_n, cold_tp, cold_fp = realized_net(cold_pred, cache.pseudo_cold_truth)

    return {
        "fold": cache.name,
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
# Load data
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
    usecols=["orderdate", "legal_entity_id", "eclass", "quantityvalue", "vk_per_item"],
    chunksize=300_000,
):
    chunk["orderdate"] = pd.to_datetime(chunk["orderdate"])
    chunks.append(chunk)

plis = pd.concat(chunks, ignore_index=True)
plis["eclass"] = pd.to_numeric(plis["eclass"], errors="coerce")
plis = plis.dropna(subset=["eclass"])
plis["eclass"] = plis["eclass"].astype(int).astype(str)
plis["line_value"] = plis["quantityvalue"] * plis["vk_per_item"]
plis["ym"] = plis["orderdate"].dt.to_period("M")
plis = plis[plis["legal_entity_id"].isin(set(customers["legal_entity_id"]))].copy()

print(f"Rows: {len(plis):,} | Warm buyers: {len(warm_ids)} | Cold buyers: {len(cold_ids)}")


# -----------------------------------------------------------------------------
# Pseudo-cold split
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Precompute fold caches (critical for Optuna speed/stability)
# -----------------------------------------------------------------------------
print("\nPreparing fold caches...")
FOLD_CACHES: list[FoldCache] = []
base_rng = np.random.RandomState(123)
for fold in ROLLING_FOLDS:
    cache = prepare_fold_cache(
        plis=plis,
        buyer_info=buyer_info,
        warm_train_ids=warm_train_ids,
        warm_eval_ids=warm_eval_ids,
        pseudo_cold_ids=pseudo_cold_ids,
        fold=fold,
        base_rng=base_rng,
    )
    FOLD_CACHES.append(cache)
print(f"Prepared {len(FOLD_CACHES)} fold caches.")


# -----------------------------------------------------------------------------
# Optuna tuning
# -----------------------------------------------------------------------------
def suggest_params(trial: optuna.Trial):
    # business knobs (high leverage)
    expected_spend_scale = trial.suggest_float("expected_spend_scale", 0.9, 1.5)
    candidate_k_cold = trial.suggest_int("candidate_k_cold", 300, 2000, step=100)

    # ranker knobs
    ranker_params = dict(BASE_RANKER_PARAMS)
    ranker_params["learning_rate"] = trial.suggest_float("rank_lr", 0.02, 0.12)
    ranker_params["num_leaves"] = trial.suggest_int("rank_num_leaves", 31, 255, log=True)
    ranker_params["min_child_samples"] = trial.suggest_int("rank_min_child", 10, 200, log=True)
    ranker_params["n_estimators"] = trial.suggest_int("rank_n_estimators", 200, 1000, step=100)

    # cold clf/reg knobs
    cold_clf_params = dict(BASE_COLD_CLF_PARAMS)
    cold_reg_params = dict(BASE_COLD_REG_PARAMS)
    cold_lr = trial.suggest_float("cold_lr", 0.02, 0.12)
    cold_leaves = trial.suggest_int("cold_num_leaves", 31, 255, log=True)
    cold_min_child = trial.suggest_int("cold_min_child", 10, 300, log=True)
    cold_n_est = trial.suggest_int("cold_n_estimators", 200, 1200, step=100)

    for p in (cold_clf_params, cold_reg_params):
        p["learning_rate"] = cold_lr
        p["num_leaves"] = cold_leaves
        p["min_child_samples"] = cold_min_child
        p["n_estimators"] = cold_n_est

    warm_clf_params = dict(BASE_WARM_CLF_PARAMS)
    warm_reg_params = dict(BASE_WARM_REG_PARAMS)

    return (
        expected_spend_scale,
        candidate_k_cold,
        warm_clf_params,
        warm_reg_params,
        ranker_params,
        cold_clf_params,
        cold_reg_params,
    )


def objective(trial: optuna.Trial) -> float:
    (
        expected_spend_scale,
        candidate_k_cold,
        warm_clf_params,
        warm_reg_params,
        ranker_params,
        cold_clf_params,
        cold_reg_params,
    ) = suggest_params(trial)

    fold_scores = []
    for i, cache in enumerate(FOLD_CACHES):
        m = evaluate_fold_cached(
            cache=cache,
            buyer_info=buyer_info,
            pseudo_cold_ids=pseudo_cold_ids,
            expected_spend_scale=expected_spend_scale,
            candidate_k_cold=candidate_k_cold,
            warm_clf_params=warm_clf_params,
            warm_reg_params=warm_reg_params,
            ranker_params=ranker_params,
            cold_clf_params=cold_clf_params,
            cold_reg_params=cold_reg_params,
        )
        fold_scores.append(m["combined_net"])

        trial.report(float(np.mean(fold_scores)), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


print("\nRunning Optuna...")
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
)
study.optimize(objective, n_trials=40)

print("\nBest trial:")
print(" value:", study.best_value)
print(" params:", study.best_params)


BEST = study.best_params
EXPECTED_SPEND_SCALE = float(BEST["expected_spend_scale"])
CANDIDATE_K_COLD = int(BEST["candidate_k_cold"])

BEST_RANKER_PARAMS = dict(BASE_RANKER_PARAMS)
BEST_RANKER_PARAMS.update(
    learning_rate=float(BEST["rank_lr"]),
    num_leaves=int(BEST["rank_num_leaves"]),
    min_child_samples=int(BEST["rank_min_child"]),
    n_estimators=int(BEST["rank_n_estimators"]),
)

BEST_COLD_CLF_PARAMS = dict(BASE_COLD_CLF_PARAMS)
BEST_COLD_REG_PARAMS = dict(BASE_COLD_REG_PARAMS)
for p in (BEST_COLD_CLF_PARAMS, BEST_COLD_REG_PARAMS):
    p.update(
        learning_rate=float(BEST["cold_lr"]),
        num_leaves=int(BEST["cold_num_leaves"]),
        min_child_samples=int(BEST["cold_min_child"]),
        n_estimators=int(BEST["cold_n_estimators"]),
    )

BEST_WARM_CLF_PARAMS = dict(BASE_WARM_CLF_PARAMS)
BEST_WARM_REG_PARAMS = dict(BASE_WARM_REG_PARAMS)

print("\nChosen tuned knobs:")
print(" EXPECTED_SPEND_SCALE:", EXPECTED_SPEND_SCALE)
print(" CANDIDATE_K_COLD:", CANDIDATE_K_COLD)


# -----------------------------------------------------------------------------
# Final training on canonical split + submission
# -----------------------------------------------------------------------------
print("\nTraining final models on canonical split...")

feat_df = plis[(plis["orderdate"] <= FEATURE_CUTOFF) & (plis["legal_entity_id"].isin(warm_ids))].copy()
label_df = plis[
    (plis["orderdate"] > FEATURE_CUTOFF) & (plis["orderdate"] <= LABEL_END) & (plis["legal_entity_id"].isin(warm_ids))
].copy()

train_feats = build_pair_features(feat_df, FEATURE_CUTOFF, buyer_info)
future_spend = (
    label_df.groupby(["legal_entity_id", "eclass"])["line_value"]
    .sum()
    .rename("future_spend")
    .reset_index()
)
train_feats = train_feats.merge(future_spend, on=["legal_entity_id", "eclass"], how="left")
train_feats["future_spend"] = train_feats["future_spend"].fillna(0.0)
train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

warm_clf, warm_reg = train_two_models_warmpair(train_feats, WARM_FEATURE_COLS, BEST_WARM_CLF_PARAMS, BEST_WARM_REG_PARAMS)

pred_cutoff = plis["orderdate"].max()
warm_full = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
warm_pred_feats = build_pair_features(warm_full, pred_cutoff, buyer_info)
warm_pred = infer_expected_spend_warmpair(warm_pred_feats, warm_clf, warm_reg, WARM_FEATURE_COLS, EXPECTED_SPEND_SCALE)

warm_sub = (
    warm_pred.loc[warm_pred["is_core"] == 1, ["legal_entity_id", "eclass"]]
    .rename(columns={"legal_entity_id": "buyer_id", "eclass": "cluster"})
)
print(f"Warm core predictions: {len(warm_sub):,}")

# Cold models: ranker + cold value models trained on warm history up to cutoff, labels until LABEL_END
hist_train = plis[(plis["orderdate"] <= FEATURE_CUTOFF) & (plis["legal_entity_id"].isin(warm_ids))].copy()
eclass_meta = build_eclass_meta(hist_train)
all_eclasses = np.array(sorted(eclass_meta["eclass"].unique().tolist()))

# Build cached-style training sets for final:
pos_pairs = hist_train.groupby(["legal_entity_id", "eclass"]).size().reset_index(name="cnt")
positives_by_buyer = {b: set(g["eclass"].tolist()) for b, g in pos_pairs.groupby("legal_entity_id")}
buyers = np.array(sorted(list(positives_by_buyer.keys())))

rng_final = np.random.RandomState(77)
sampled_rank = sample_negative_pairs(buyers, positives_by_buyer, all_eclasses, NEG_PER_POS_RANK, rng_final)
rank_train_feats = make_cold_rank_features(sampled_rank, buyer_info, eclass_meta)
rank_group_sizes = rank_train_feats.groupby("legal_entity_id").size().loc[buyers].tolist()

ranker = train_ranker_from_cached(rank_train_feats, rank_group_sizes, BEST_RANKER_PARAMS)

# cold value train based on future spend (cutoff+1..LABEL_END)
truth_train = build_future_spend_labels(plis, FEATURE_CUTOFF + pd.Timedelta(days=1), LABEL_END, warm_ids)
positives_future = {b: set(g.loc[g["future_spend"] > 0, "eclass"].tolist()) for b, g in truth_train.groupby("legal_entity_id")}
buyers2 = np.array(sorted(list(positives_future.keys())))

sampled_cold = sample_negative_pairs(buyers2, positives_future, all_eclasses, NEG_PER_POS_COLD, np.random.RandomState(88))
sampled_cold = sampled_cold.merge(truth_train, on=["legal_entity_id", "eclass"], how="left")
sampled_cold["future_spend"] = sampled_cold["future_spend"].fillna(0.0)

cold_train_feats = make_cold_rank_features(sampled_cold, buyer_info, eclass_meta)
cold_train_feats["log_spend_label"] = np.log1p(cold_train_feats["future_spend"])

cold_clf, cold_reg = train_cold_value_models_from_cached(cold_train_feats, ranker, BEST_COLD_CLF_PARAMS, BEST_COLD_REG_PARAMS)

# predict cold
cold_list = sorted(list(cold_ids))
if len(cold_list) == 0:
    cold_sub = pd.DataFrame(columns=["buyer_id", "cluster"])
    print("Cold: no cold buyers")
else:
    cold_cand = score_candidates_for_buyers_batch(
        buyer_ids=cold_list,
        all_eclasses=all_eclasses,
        buyer_info=buyer_info,
        eclass_meta=eclass_meta,
        ranker=ranker,
        top_k=CANDIDATE_K_COLD,
    )
    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cold_cand,
        buyer_info=buyer_info,
        eclass_meta=eclass_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=EXPECTED_SPEND_SCALE,
    )
    cold_sub = (
        cold_pred.loc[cold_pred["is_core"] == 1, ["legal_entity_id", "eclass"]]
        .rename(columns={"legal_entity_id": "buyer_id", "eclass": "cluster"})
    )
    covered = cold_sub["buyer_id"].nunique() if not cold_sub.empty else 0
    print(f"Cold core predictions: {len(cold_sub):,} across {covered}/{len(cold_ids)} buyers")

# submission
submission = (
    pd.concat([warm_sub, cold_sub], ignore_index=True)
    .drop_duplicates()
    .sort_values(["buyer_id", "cluster"])
    .reset_index(drop=True)
)
submission.to_csv(OUT / "submission.csv", index=False)
print(f"Saved submission.csv with {len(submission):,} rows ({len(warm_sub)} warm + {len(cold_sub)} cold)")
