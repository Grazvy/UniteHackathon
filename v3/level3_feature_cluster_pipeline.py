"""Level 3 pipeline: E-Class + clustered feature combinations.

This script implements a practical Level-3 approach on top of the existing two-stage
(candidates + value) pipeline used in Level 1/2.

Level 3 idea
- Build a stable `cluster_id` per product need defined as (eclass + feature-combination).
- Feature-combination is derived from `features_per_sku.csv` (long format attributes).
- Cluster deterministically by (within-eclass) feature-profiles, using a small set of
  high-support feature keys per eclass and hashing the resulting signature.

Cold/warm handling
- Warm buyers: candidates are historical (buyer, cluster_id) pairs with pair-history features.
- Cold buyers: candidates are top-K cluster_ids from a ranker trained on buyer metadata +
  cluster popularity/meta features.

Output
- Writes `submission_level3.csv` to repo root with columns: buyer_id,predicted_id

Notes
- This is intentionally deterministic and dependency-light (no scikit-learn clustering).
- Some SKUs may have missing/absent features; they fall back to an eclass-only cluster.
"""

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from hashlib import sha1
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb


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
CANDIDATE_K_COLD = 800
NEG_PER_POS = 40

# LightGBM Lambdarank has a hard-ish group size limit (~10k rows per query).
MAX_GROUP_SIZE_RANKER = 9000
MAX_POS_PER_BUYER_RANKER = 3000

EXPECTED_SPEND_SCALE_GRID = [1.3]
CANDIDATE_K_COLD_GRID = [800]

# Level-3 clustering controls
TOP_KEYS_PER_ECLASS = 6
MIN_KEY_SKU_SUPPORT = 0.05  # key must appear on >=5% of SKUs in that eclass
MAX_KEYS_PER_SKU = 20  # hard safety cap
CLUSTER_HASH_LEN = 10

TARGET_ID_COL = "cluster_id"
PAIR_KEY_COLS = ["legal_entity_id", TARGET_ID_COL]

LGBM_PARAMS_WARM_CLF = {
    "n_estimators": 350,
    "max_depth": 6,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 25,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}

LGBM_PARAMS_WARM_REG = {
    "n_estimators": 350,
    "max_depth": 6,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 25,
    "random_state": 42,
    "verbose": -1,
}

LGBM_PARAMS_RANKER = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

LGBM_PARAMS_COLD_CLF = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 25,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}

LGBM_PARAMS_COLD_REG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 25,
    "random_state": 42,
    "verbose": -1,
}

DATA = Path(__file__).resolve().parent.parent / "data" / "unite"
OUT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Helpers: cleaning + hashing
# -----------------------------------------------------------------------------

def _clean_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = " ".join(s.split())
    return s


def _clean_key(x: object) -> str:
    return _clean_text(x).casefold()


def _clean_val(x: object) -> str:
    return _clean_text(x).casefold()


def _hash_signature(signature: str) -> str:
    return sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:CLUSTER_HASH_LEN]


@dataclass(frozen=True)
class ClusterConfig:
    top_keys_per_eclass: int = TOP_KEYS_PER_ECLASS
    min_key_sku_support: float = MIN_KEY_SKU_SUPPORT
    max_keys_per_sku: int = MAX_KEYS_PER_SKU


# -----------------------------------------------------------------------------
# Level-3 clustering
# -----------------------------------------------------------------------------

def build_sku_to_eclass_map(plis_path: Path, chunksize: int = 400_000) -> pd.DataFrame:
    """Build a sku->eclass mapping by streaming plis_training.

    Assumes each SKU maps to a single eclass (or we take the first observed).
    """
    mapping: dict[str, str] = {}
    for chunk in pd.read_csv(
        plis_path,
        sep="\t",
        usecols=["sku", "eclass"],
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["sku", "eclass"])
        chunk["eclass"] = pd.to_numeric(chunk["eclass"], errors="coerce")
        chunk = chunk.dropna(subset=["eclass"])
        chunk["eclass"] = chunk["eclass"].astype(int).astype(str)
        # Keep first mapping only
        for sku, eclass in zip(chunk["sku"].astype(str).tolist(), chunk["eclass"].tolist()):
            if sku not in mapping:
                mapping[sku] = eclass
    return pd.DataFrame({"sku": list(mapping.keys()), "eclass": list(mapping.values())})


def build_feature_profiles(
    features_path: Path,
    sku_to_eclass: pd.DataFrame,
    cfg: ClusterConfig,
    chunksize: int = 600_000,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build per-sku feature profiles and select top keys per eclass.

    Returns:
    - sku_profile: DataFrame with columns [sku, eclass, safe_synonym, profile_signature]
    - eclass_top_keys: dict[eclass] -> list[key]

    Implementation details:
    - Uses `fvalue_set` when available, else falls back to `fvalue`.
    - For each (sku, key), chooses the modal value over rows.
    - Selects top keys per eclass by #distinct skus that have the key.
    """

    sku_to_eclass_map = dict(zip(sku_to_eclass["sku"].astype(str), sku_to_eclass["eclass"].astype(str)))

    # Accumulators
    # 1) counts for picking modal (sku,key)->value
    value_counts: dict[tuple[str, str, str], int] = {}
    # 2) key support per eclass: count distinct skus per (eclass, key_norm)
    seen_eclass_key_sku: set[tuple[str, str, str]] = set()
    key_sku_counts: dict[tuple[str, str], int] = {}

    safe_synonym_by_sku: dict[str, str] = {}

    for chunk in pd.read_csv(
        features_path,
        sep="\t",
        usecols=["safe_synonym", "sku", "key", "fvalue", "fvalue_set"],
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["sku", "key"])
        chunk["sku"] = chunk["sku"].astype(str)

        # Map to eclass; if SKU never appears in plis_training, skip (not used for training/pred)
        chunk["eclass"] = chunk["sku"].map(sku_to_eclass_map)
        chunk = chunk.dropna(subset=["eclass"]).copy()
        if chunk.empty:
            continue

        chunk["safe_synonym"] = chunk["safe_synonym"].astype(str).fillna("")
        # Prefer fvalue_set
        val = chunk["fvalue_set"].where(chunk["fvalue_set"].notna(), chunk["fvalue"])
        chunk["val"] = val.astype(str).fillna("")

        # Normalize
        chunk["key_norm"] = chunk["key"].map(_clean_key)
        chunk["val_norm"] = chunk["val"].map(_clean_val)

        # Safety: drop empty keys/values
        chunk = chunk[(chunk["key_norm"] != "") & (chunk["val_norm"] != "")]
        if chunk.empty:
            continue

        # Keep a deterministic safe_synonym per sku (first seen)
        for sku, ss in zip(chunk["sku"].tolist(), chunk["safe_synonym"].tolist()):
            if sku not in safe_synonym_by_sku:
                safe_synonym_by_sku[sku] = _clean_text(ss)

        # Update key support counts (distinct sku per (eclass,key_norm))
        # Use vectorized unique pairs per chunk, then update python dict
        unique_pairs = chunk[["eclass", "sku", "key_norm"]].drop_duplicates()
        for eclass, sku, key_norm in zip(
            unique_pairs["eclass"].tolist(),
            unique_pairs["sku"].tolist(),
            unique_pairs["key_norm"].tolist(),
        ):
            sk = (eclass, key_norm)
            esk = (str(eclass), str(key_norm), str(sku))
            if esk not in seen_eclass_key_sku:
                seen_eclass_key_sku.add(esk)
                key_sku_counts[sk] = key_sku_counts.get(sk, 0) + 1

        # Update value counts for (sku,key)->modal value
        grouped = (
            chunk.groupby(["sku", "key_norm", "val_norm"], as_index=False)
            .size()
            .rename(columns={"size": "cnt"})
        )
        for sku, key_norm, val_norm, cnt in zip(
            grouped["sku"].tolist(),
            grouped["key_norm"].tolist(),
            grouped["val_norm"].tolist(),
            grouped["cnt"].tolist(),
        ):
            k = (sku, key_norm, val_norm)
            value_counts[k] = value_counts.get(k, 0) + int(cnt)

    # Build (eclass -> top keys)
    key_counts_df = pd.DataFrame(
        [(eclass, key_norm, cnt) for (eclass, key_norm), cnt in key_sku_counts.items()],
        columns=["eclass", "key_norm", "sku_count"],
    )
    if key_counts_df.empty:
        raise RuntimeError("No usable feature keys found; check features_per_sku.csv parsing.")

    eclass_sku_counts = (
        sku_to_eclass.groupby("eclass")["sku"].nunique().rename("n_skus").reset_index()
    )
    key_counts_df = key_counts_df.merge(eclass_sku_counts, on="eclass", how="left")
    key_counts_df["support"] = key_counts_df["sku_count"] / key_counts_df["n_skus"].clip(lower=1)
    key_counts_df = key_counts_df[key_counts_df["support"] >= cfg.min_key_sku_support]

    eclass_top_keys: dict[str, list[str]] = {}
    for eclass, g in key_counts_df.sort_values(["eclass", "sku_count"], ascending=[True, False]).groupby("eclass"):
        eclass_top_keys[str(eclass)] = g["key_norm"].head(cfg.top_keys_per_eclass).tolist()

    # Choose modal value per (sku, key_norm)
    vc_df = pd.DataFrame(
        [(sku, key_norm, val_norm, cnt) for (sku, key_norm, val_norm), cnt in value_counts.items()],
        columns=["sku", "key_norm", "val_norm", "cnt"],
    )
    if vc_df.empty:
        # No values at all -> every SKU falls back
        sku_profile = sku_to_eclass.copy()
        sku_profile["safe_synonym"] = sku_profile["sku"].astype(str).map(safe_synonym_by_sku).fillna("")
        sku_profile["profile_signature"] = "__NOFEATURES__"
        return sku_profile, eclass_top_keys

    # Reduce to (sku,key)->best value
    vc_df = vc_df.sort_values(["sku", "key_norm", "cnt"], ascending=[True, True, False])
    best = vc_df.drop_duplicates(subset=["sku", "key_norm"], keep="first")

    # Build signature per sku
    # Join eclass for key selection
    best["eclass"] = best["sku"].map(sku_to_eclass_map)
    best = best.dropna(subset=["eclass"]).copy()

    # Group to dict per sku
    profiles: dict[str, dict[str, str]] = {}
    for sku, eclass, key_norm, val_norm in zip(
        best["sku"].tolist(),
        best["eclass"].astype(str).tolist(),
        best["key_norm"].tolist(),
        best["val_norm"].tolist(),
    ):
        keys = eclass_top_keys.get(str(eclass), [])
        if key_norm not in keys:
            continue
        d = profiles.get(sku)
        if d is None:
            d = {}
            profiles[sku] = d
        if len(d) >= cfg.max_keys_per_sku:
            continue
        d[key_norm] = val_norm

    sku_profile = sku_to_eclass.copy()
    sku_profile["safe_synonym"] = sku_profile["sku"].astype(str).map(safe_synonym_by_sku).fillna("")

    def _sig(row: pd.Series) -> str:
        sku = str(row["sku"])
        eclass = str(row["eclass"])
        keys = eclass_top_keys.get(eclass, [])
        d = profiles.get(sku, {})
        parts = []
        for k in keys:
            v = d.get(k)
            if v is None or v == "":
                continue
            parts.append(f"{k}={v}")
        if not parts:
            # Use safe_synonym if present as a weak fallback before eclass-only
            ss = _clean_key(row.get("safe_synonym", ""))
            if ss:
                return f"safe_synonym={ss}"
            return "__NOFEATURES__"
        return ";".join(sorted(parts))

    sku_profile["profile_signature"] = sku_profile.apply(_sig, axis=1)
    return sku_profile, eclass_top_keys


def build_cluster_map(
    sku_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Map each (sku,eclass) to a stable cluster_id = eclass|hash(signature)."""
    sig = sku_profile["profile_signature"].fillna("__NOFEATURES__").astype(str)
    sku_profile = sku_profile.copy()
    sku_profile["cluster_code"] = sig.map(_hash_signature)
    sku_profile[TARGET_ID_COL] = sku_profile["eclass"].astype(str) + "|" + sku_profile["cluster_code"].astype(str)
    return sku_profile[["sku", "eclass", TARGET_ID_COL]].drop_duplicates(subset=["sku"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Warm pair-history features (buyer, cluster_id)
# -----------------------------------------------------------------------------

def build_pair_features(df: pd.DataFrame, cutoff: pd.Timestamp, buyer_info: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(PAIR_KEY_COLS)
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
        .groupby(PAIR_KEY_COLS)["line_value"]
        .sum()
        .rename("last6m_spend")
        .reset_index()
    )
    feats = feats.merge(last6, on=PAIR_KEY_COLS, how="left")
    feats["last6m_spend"] = feats["last6m_spend"].fillna(0)

    buyer_total = df.groupby("legal_entity_id")["line_value"].sum().rename("buyer_total_spend").reset_index()
    cluster_total = df.groupby(TARGET_ID_COL)["line_value"].sum().rename("cluster_total_spend").reset_index()

    feats = feats.merge(buyer_total, on="legal_entity_id", how="left")
    feats = feats.merge(cluster_total, on=TARGET_ID_COL, how="left")
    feats = feats.merge(
        buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
        on="legal_entity_id",
        how="left",
    )

    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        feats[c] = feats[c].fillna(-1)
    feats["cluster_total_spend"] = feats["cluster_total_spend"].fillna(0)

    return feats.drop(columns=["last_order", "first_order"])


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


def train_two_models_warmpair(train_feats: pd.DataFrame, feature_cols: list[str]):
    X = train_feats[feature_cols].fillna(0)
    clf = lgb.LGBMClassifier(**LGBM_PARAMS_WARM_CLF)
    clf.fit(X, train_feats["buy_label"])

    pos_mask = train_feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS_WARM_REG)
    reg.fit(X.loc[pos_mask], train_feats.loc[pos_mask, "log_spend_label"])
    return clf, reg


def realized_net(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, int, int, int]:
    merged = pred_df[["legal_entity_id", TARGET_ID_COL, "is_core"]].merge(
        truth_df[["legal_entity_id", TARGET_ID_COL, "future_spend"]],
        on=["legal_entity_id", TARGET_ID_COL],
        how="left",
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
# Cold-start models
# -----------------------------------------------------------------------------

def build_cluster_meta(plis_hist: pd.DataFrame) -> pd.DataFrame:
    meta = plis_hist.groupby(TARGET_ID_COL, as_index=False).agg(
        cluster_total_spend=("line_value", "sum"),
        cluster_n_orders=("line_value", "size"),
        cluster_n_buyers=("legal_entity_id", "nunique"),
    )
    meta["log_cluster_total_spend"] = np.log1p(meta["cluster_total_spend"])
    meta["log_cluster_n_orders"] = np.log1p(meta["cluster_n_orders"])
    meta["log_cluster_n_buyers"] = np.log1p(meta["cluster_n_buyers"])
    return meta.drop_duplicates(subset=[TARGET_ID_COL]).reset_index(drop=True)


def sample_negative_pairs(
    buyers: np.ndarray,
    positives_by_buyer: dict[int, set[str]],
    all_targets: np.ndarray,
    n_neg_per_pos: int,
    rng: np.random.RandomState,
    max_group_size: int = MAX_GROUP_SIZE_RANKER,
    max_pos_per_buyer: int | None = MAX_POS_PER_BUYER_RANKER,
) -> pd.DataFrame:
    rows: list[tuple[int, str, int]] = []
    all_set = set(all_targets.tolist())

    for b in buyers:
        pos = positives_by_buyer.get(int(b), set())
        if not pos:
            continue

        pos_list = list(pos)
        if max_pos_per_buyer is not None and len(pos_list) > max_pos_per_buyer:
            pos_list = rng.choice(pos_list, size=max_pos_per_buyer, replace=False).tolist()

        n_pos = len(pos_list)
        if n_pos == 0:
            continue

        max_neg_allowed = max_group_size - n_pos
        if max_neg_allowed <= 0:
            for t in pos_list:
                rows.append((int(b), t, 1))
            continue

        target_neg = min(n_pos * n_neg_per_pos, len(all_set) - len(pos), max_neg_allowed)
        if target_neg <= 0:
            for t in pos_list:
                rows.append((int(b), t, 1))
            continue

        cand = rng.choice(all_targets, size=target_neg * 3, replace=True).tolist()
        negs: list[str] = []
        for t in cand:
            if t not in pos:
                negs.append(t)
                if len(negs) >= target_neg:
                    break

        for t in pos_list:
            rows.append((int(b), t, 1))
        for t in negs:
            rows.append((int(b), t, 0))

    return pd.DataFrame(rows, columns=["legal_entity_id", TARGET_ID_COL, "label"])


COLD_FEATS_NUM = [
    "log_employees",
    "section_enc",
    "nace_2digits",
    "has_secondary_nace",
    "log_cluster_total_spend",
    "log_cluster_n_orders",
    "log_cluster_n_buyers",
]
COLD_FEATS_CAT = ["cluster_cat"]


def make_cold_rank_features(
    pairs: pd.DataFrame,
    buyer_info: pd.DataFrame,
    cluster_meta: pd.DataFrame,
) -> pd.DataFrame:
    df = pairs.merge(
        buyer_info[["legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]],
        on="legal_entity_id",
        how="left",
    ).merge(
        cluster_meta[[TARGET_ID_COL, "log_cluster_total_spend", "log_cluster_n_orders", "log_cluster_n_buyers"]],
        on=TARGET_ID_COL,
        how="left",
    )

    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        df[c] = df[c].fillna(-1)
    for c in ["log_cluster_total_spend", "log_cluster_n_orders", "log_cluster_n_buyers"]:
        df[c] = df[c].fillna(0)

    df["cluster_cat"] = df[TARGET_ID_COL].astype("category")

    if df.duplicated(subset=["legal_entity_id", TARGET_ID_COL]).any():
        df = df.drop_duplicates(subset=["legal_entity_id", TARGET_ID_COL]).reset_index(drop=True)
    return df


def train_ranker(
    plis_hist: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set[int],
    train_end: pd.Timestamp,
    rng: np.random.RandomState,
    neg_per_pos: int = NEG_PER_POS,
):
    hist = plis_hist[(plis_hist["orderdate"] <= train_end) & (plis_hist["legal_entity_id"].isin(warm_train_ids))].copy()
    if hist.empty:
        return None, None

    cluster_meta = build_cluster_meta(hist)
    pos_pairs = hist.groupby(["legal_entity_id", TARGET_ID_COL]).size().reset_index(name="cnt")
    positives_by_buyer: dict[int, set[str]] = {}
    for buyer_id, g in pos_pairs.groupby("legal_entity_id"):
        positives_by_buyer[int(buyer_id)] = set(g[TARGET_ID_COL].astype(str).tolist())

    all_targets = np.array(sorted(cluster_meta[TARGET_ID_COL].astype(str).tolist()))
    buyers = pos_pairs["legal_entity_id"].drop_duplicates().sort_values().to_numpy(dtype=np.int64)

    sampled = sample_negative_pairs(
        buyers=buyers,
        positives_by_buyer=positives_by_buyer,
        all_targets=all_targets,
        n_neg_per_pos=neg_per_pos,
        rng=rng,
    )
    feats = make_cold_rank_features(sampled, buyer_info, cluster_meta)

    X = feats[COLD_FEATS_NUM + COLD_FEATS_CAT]
    y = feats["label"].astype(int)
    group_sizes = feats.groupby("legal_entity_id").size().loc[buyers].tolist()

    ranker = lgb.LGBMRanker(**LGBM_PARAMS_RANKER)
    ranker.fit(X, y, group=group_sizes, categorical_feature=COLD_FEATS_CAT)
    return ranker, cluster_meta


def score_candidates_for_buyers(
    buyer_ids: list[int],
    cluster_meta: pd.DataFrame,
    buyer_info: pd.DataFrame,
    ranker,
    top_k: int,
) -> pd.DataFrame:
    universe = cluster_meta[[TARGET_ID_COL]].drop_duplicates().reset_index(drop=True)
    rows = []
    for b in buyer_ids:
        tmp = universe.copy()
        tmp["legal_entity_id"] = int(b)
        tmp["label"] = 0
        feats = make_cold_rank_features(tmp, buyer_info, cluster_meta)
        feats["rank_score"] = ranker.predict(feats[COLD_FEATS_NUM + COLD_FEATS_CAT])
        feats = feats.sort_values("rank_score", ascending=False).head(top_k)
        rows.append(feats[["legal_entity_id", TARGET_ID_COL, "rank_score"]])

    if not rows:
        return pd.DataFrame(columns=["legal_entity_id", TARGET_ID_COL, "rank_score"])
    return pd.concat(rows, ignore_index=True)


def build_future_spend_labels(
    plis: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    buyer_ids: set[int],
) -> pd.DataFrame:
    lab = plis[(plis["orderdate"] >= start) & (plis["orderdate"] <= end) & (plis["legal_entity_id"].isin(buyer_ids))].copy()
    return (
        lab.groupby(PAIR_KEY_COLS)["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )


def train_cold_value_models(
    plis_hist: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set[int],
    fold: dict,
    ranker,
    cluster_meta: pd.DataFrame,
    rng: np.random.RandomState,
    neg_per_pos: int = NEG_PER_POS,
):
    train_end, val_start, val_end = fold["train_end"], fold["val_start"], fold["val_end"]

    hist = plis_hist[(plis_hist["orderdate"] <= train_end) & (plis_hist["legal_entity_id"].isin(warm_train_ids))].copy()
    if hist.empty:
        return None, None

    truth = build_future_spend_labels(plis_hist, val_start, val_end, warm_train_ids)
    positives_by_buyer: dict[int, set[str]] = {}
    for buyer_id, g in truth.groupby("legal_entity_id"):
        positives_by_buyer[int(buyer_id)] = set(
            g.loc[g["future_spend"] > 0, TARGET_ID_COL].astype(str).tolist()
        )
    buyers = truth["legal_entity_id"].drop_duplicates().sort_values().to_numpy(dtype=np.int64)
    if len(buyers) == 0:
        return None, None

    all_targets = np.array(sorted(cluster_meta[TARGET_ID_COL].astype(str).tolist()))
    sampled = sample_negative_pairs(
        buyers=buyers,
        positives_by_buyer=positives_by_buyer,
        all_targets=all_targets,
        n_neg_per_pos=neg_per_pos,
        rng=rng,
        max_group_size=MAX_GROUP_SIZE_RANKER,
        max_pos_per_buyer=MAX_POS_PER_BUYER_RANKER,
    )
    sampled = sampled.merge(
        truth[["legal_entity_id", TARGET_ID_COL, "future_spend"]],
        on=["legal_entity_id", TARGET_ID_COL],
        how="left",
    )
    sampled["future_spend"] = sampled["future_spend"].fillna(0.0)

    feats = make_cold_rank_features(sampled, buyer_info, cluster_meta)
    feats["rank_score"] = ranker.predict(feats[COLD_FEATS_NUM + COLD_FEATS_CAT])
    feats["log_spend_label"] = np.log1p(feats["future_spend"])

    cold_value_feats = COLD_FEATS_NUM + ["rank_score"] + COLD_FEATS_CAT

    clf = lgb.LGBMClassifier(**LGBM_PARAMS_COLD_CLF)
    clf.fit(
        feats[cold_value_feats],
        (feats["future_spend"] > 0).astype(int),
        categorical_feature=COLD_FEATS_CAT,
    )

    pos_mask = feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS_COLD_REG)
    if pos_mask.sum() > 20:
        reg.fit(
            feats.loc[pos_mask, cold_value_feats],
            feats.loc[pos_mask, "log_spend_label"],
            categorical_feature=COLD_FEATS_CAT,
        )
    else:
        reg.fit(
            feats[cold_value_feats].head(50),
            np.zeros(min(50, len(feats))),
            categorical_feature=COLD_FEATS_CAT,
        )

    return clf, reg


def infer_expected_spend_cold(
    candidate_pairs: pd.DataFrame,
    buyer_info: pd.DataFrame,
    cluster_meta: pd.DataFrame,
    ranker,
    cold_clf,
    cold_reg,
    expected_spend_scale: float,
) -> pd.DataFrame:
    base = candidate_pairs.copy()
    if "rank_score" not in base.columns:
        base["rank_score"] = np.nan

    feats = make_cold_rank_features(base.assign(label=0), buyer_info, cluster_meta)
    if "rank_score" not in feats.columns:
        feats = feats.merge(
            base[["legal_entity_id", TARGET_ID_COL, "rank_score"]],
            on=["legal_entity_id", TARGET_ID_COL],
            how="left",
            suffixes=("", "_base"),
        )

    needs_score = feats["rank_score"].isna()
    if needs_score.any():
        feats.loc[needs_score, "rank_score"] = ranker.predict(
            feats.loc[needs_score, COLD_FEATS_NUM + COLD_FEATS_CAT]
        )

    cold_value_feats = COLD_FEATS_NUM + ["rank_score"] + COLD_FEATS_CAT
    X = feats[cold_value_feats]

    p_buy = cold_clf.predict_proba(X)[:, 1]
    spend_if_buy = np.expm1(cold_reg.predict(X)).clip(min=0)

    out = feats[["legal_entity_id", TARGET_ID_COL, "rank_score"]].copy()
    out["p_buy"] = p_buy
    out["spend_if_buy"] = spend_if_buy
    out["expected_spend"] = p_buy * spend_if_buy * expected_spend_scale
    out["expected_savings"] = out["expected_spend"] * SAVINGS_RATE
    out["is_core"] = (out["expected_savings"] > TOTAL_FEE).astype(int)
    return out


# -----------------------------------------------------------------------------
# Rolling evaluation
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
    "cluster_total_spend",
    "log_employees",
    "section_enc",
    "nace_2digits",
    "has_secondary_nace",
]


def evaluate_fold(
    plis: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set[int],
    warm_eval_ids: set[int],
    pseudo_cold_ids: set[int],
    fold: dict,
    expected_spend_scale: float,
    candidate_k_cold: int,
    rng: np.random.RandomState,
) -> dict:
    train_end, val_start, val_end = fold["train_end"], fold["val_start"], fold["val_end"]

    feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_train_ids))].copy()
    label_df = plis[
        (plis["orderdate"] >= val_start)
        & (plis["orderdate"] <= val_end)
        & (plis["legal_entity_id"].isin(warm_train_ids.union(warm_eval_ids).union(pseudo_cold_ids)))
    ].copy()

    train_feats = build_pair_features(feat_df, train_end, buyer_info)
    future_spend_train = (
        label_df[label_df["legal_entity_id"].isin(warm_train_ids)]
        .groupby(PAIR_KEY_COLS)["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    train_feats = train_feats.merge(future_spend_train, on=PAIR_KEY_COLS, how="left")
    train_feats["future_spend"] = train_feats["future_spend"].fillna(0)
    train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
    train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

    if train_feats.empty:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    warm_clf, warm_reg = train_two_models_warmpair(train_feats, WARM_FEATURE_COLS)
    ranker, cluster_meta = train_ranker(plis, buyer_info, warm_train_ids, train_end, rng=rng, neg_per_pos=NEG_PER_POS)
    if ranker is None or cluster_meta is None:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    cold_clf, cold_reg = train_cold_value_models(
        plis_hist=plis,
        buyer_info=buyer_info,
        warm_train_ids=warm_train_ids,
        fold=fold,
        ranker=ranker,
        cluster_meta=cluster_meta,
        rng=rng,
        neg_per_pos=NEG_PER_POS,
    )
    if cold_clf is None or cold_reg is None:
        return {"fold": fold["name"], "warm_net": -np.inf, "cold_proxy_net": -np.inf, "combined_net": -np.inf}

    warm_eval_feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_eval_ids))].copy()
    warm_eval_feats = build_pair_features(warm_eval_feat_df, train_end, buyer_info)
    warm_eval_pred = infer_expected_spend_warmpair(
        warm_eval_feats,
        warm_clf,
        warm_reg,
        WARM_FEATURE_COLS,
        expected_spend_scale,
    )

    warm_truth = (
        label_df[label_df["legal_entity_id"].isin(warm_eval_ids)]
        .groupby(PAIR_KEY_COLS)["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_eval_pred, warm_truth)

    pseudo_cold_list = sorted(list(pseudo_cold_ids))
    cand = score_candidates_for_buyers(
        buyer_ids=pseudo_cold_list,
        cluster_meta=cluster_meta,
        buyer_info=buyer_info,
        ranker=ranker,
        top_k=candidate_k_cold,
    )
    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cand,
        buyer_info=buyer_info,
        cluster_meta=cluster_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=expected_spend_scale,
    )
    cold_truth = (
        label_df[label_df["legal_entity_id"].isin(pseudo_cold_ids)]
        .groupby(PAIR_KEY_COLS)["line_value"]
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
# Main
# -----------------------------------------------------------------------------
print("Loading buyers...")

print(
    "Assumptions: "
    f"SAVINGS_RATE={SAVINGS_RATE:.3f}, FEE={TOTAL_FEE:.2f} per element-month, "
    f"decision rule: expected_spend * SAVINGS_RATE > FEE (i.e. expected_spend > {TOTAL_FEE / max(SAVINGS_RATE, 1e-9):.2f})."
)
customers = pd.read_csv(DATA / "customer_test.csv", sep="\t")
customers["nace_code"] = customers["nace_code"].astype(int)

nace = pd.read_csv(DATA / "nace_codes.csv", sep="\t")
nace["nace_code"] = nace["nace_code"].astype(int)

sections = sorted(nace["toplevel_section"].dropna().unique())
section_map = {section: idx for idx, section in enumerate(sections)}

buyer_info = customers.merge(
    nace[["nace_code", "toplevel_section", "nace_2digits"]],
    on="nace_code",
    how="left",
)
buyer_info["section_enc"] = buyer_info["toplevel_section"].map(section_map).fillna(-1).astype(int)
buyer_info["nace_2digits"] = buyer_info["nace_2digits"].fillna(-1)
buyer_info["has_secondary_nace"] = buyer_info["secondary_nace_code"].notna().astype(int)
buyer_info["log_employees"] = np.log1p(buyer_info["estimated_number_employees"].fillna(0))

warm_ids = set(customers.loc[customers["task"] == "predict future", "legal_entity_id"].astype(int))
cold_ids = set(customers.loc[customers["task"] == "cold start", "legal_entity_id"].astype(int))

print("Building Level-3 clusters (sku -> cluster_id)...")
sku_to_eclass = build_sku_to_eclass_map(DATA / "plis_training.csv")
profiles, _ = build_feature_profiles(DATA / "features_per_sku.csv", sku_to_eclass, cfg=ClusterConfig())
sku_cluster_map = build_cluster_map(profiles)

print("Loading transactions + attaching cluster_id...")
chunks = []
for chunk in pd.read_csv(
    DATA / "plis_training.csv",
    sep="\t",
    low_memory=False,
    usecols=["orderdate", "legal_entity_id", "sku", "quantityvalue", "vk_per_item"],
    chunksize=300_000,
):
    chunk["orderdate"] = pd.to_datetime(chunk["orderdate"])
    chunk["legal_entity_id"] = chunk["legal_entity_id"].astype(int)
    chunk["sku"] = chunk["sku"].astype(str)
    chunk = chunk.merge(sku_cluster_map, on="sku", how="left")
    # If SKU has no features mapping, fall back to eclass-only cluster when possible.
    fallback_code = _hash_signature("__NOFEATURES__")
    has_eclass = chunk["eclass"].notna()
    chunk.loc[chunk[TARGET_ID_COL].isna() & has_eclass, TARGET_ID_COL] = (
        chunk.loc[chunk[TARGET_ID_COL].isna() & has_eclass, "eclass"].astype(str) + "|" + fallback_code
    )
    chunk[TARGET_ID_COL] = chunk[TARGET_ID_COL].fillna("__MISSING_CLUSTER__")

    chunk["line_value"] = chunk["quantityvalue"] * chunk["vk_per_item"]
    chunk["ym"] = chunk["orderdate"].dt.to_period("M")
    chunk = chunk[chunk["legal_entity_id"].isin(set(customers["legal_entity_id"].astype(int)))].copy()
    chunks.append(chunk[["orderdate", "legal_entity_id", TARGET_ID_COL, "line_value", "ym"]])

plis = pd.concat(chunks, ignore_index=True)
print(
    f"Rows: {len(plis):,} | Clusters: {plis[TARGET_ID_COL].nunique():,} | "
    f"Warm buyers: {len(warm_ids)} | Cold buyers: {len(cold_ids)}"
)


# 2) Rolling validation + net tuning
print("\nRunning rolling validation + net-score tuning...")
rng = np.random.RandomState(42)
warm_list = np.array(sorted(map(int, warm_ids)))
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


# 3) Final training on canonical split
print("\nTraining final models on canonical split...")
feat_df = plis[(plis["orderdate"] <= FEATURE_CUTOFF) & (plis["legal_entity_id"].isin(warm_ids))].copy()
label_df = plis[(plis["orderdate"] > FEATURE_CUTOFF) & (plis["orderdate"] <= LABEL_END) & (plis["legal_entity_id"].isin(warm_ids))].copy()

train_feats = build_pair_features(feat_df, FEATURE_CUTOFF, buyer_info)
future_spend = (
    label_df.groupby(PAIR_KEY_COLS)["line_value"]
    .sum()
    .rename("future_spend")
    .reset_index()
)
train_feats = train_feats.merge(future_spend, on=PAIR_KEY_COLS, how="left")
train_feats["future_spend"] = train_feats["future_spend"].fillna(0)
train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

warm_clf, warm_reg = train_two_models_warmpair(train_feats, WARM_FEATURE_COLS)

final_rng = np.random.RandomState(42)
ranker, cluster_meta = train_ranker(plis, buyer_info, warm_ids, FEATURE_CUTOFF, rng=final_rng, neg_per_pos=NEG_PER_POS)
if ranker is None or cluster_meta is None:
    raise RuntimeError("Final ranker training failed; no Level-3 target universe available.")

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
    cluster_meta=cluster_meta,
    rng=np.random.RandomState(7),
    neg_per_pos=NEG_PER_POS,
)


# 4) Final predictions: warm
pred_cutoff = plis["orderdate"].max()

warm_full = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
warm_pred_feats = build_pair_features(warm_full, pred_cutoff, buyer_info)
warm_pred = infer_expected_spend_warmpair(
    warm_pred_feats,
    warm_clf,
    warm_reg,
    WARM_FEATURE_COLS,
    EXPECTED_SPEND_SCALE,
)

warm_sub = (
    warm_pred.loc[warm_pred["is_core"] == 1, ["legal_entity_id", TARGET_ID_COL]]
    .rename(columns={"legal_entity_id": "buyer_id", TARGET_ID_COL: "predicted_id"})
)
print(f"Warm core predictions: {len(warm_sub):,}")


# 5) Final predictions: cold
cold_list = sorted(map(int, cold_ids))
if len(cold_list) == 0:
    cold_sub = pd.DataFrame(columns=["buyer_id", "predicted_id"])
    print("Cold: no cold buyers")
else:
    cold_cand = score_candidates_for_buyers(
        buyer_ids=cold_list,
        cluster_meta=cluster_meta,
        buyer_info=buyer_info,
        ranker=ranker,
        top_k=CANDIDATE_K_COLD,
    )
    cold_pred = infer_expected_spend_cold(
        candidate_pairs=cold_cand,
        buyer_info=buyer_info,
        cluster_meta=cluster_meta,
        ranker=ranker,
        cold_clf=cold_clf,
        cold_reg=cold_reg,
        expected_spend_scale=EXPECTED_SPEND_SCALE,
    )

    cold_sub = (
        cold_pred.loc[cold_pred["is_core"] == 1, ["legal_entity_id", TARGET_ID_COL]]
        .rename(columns={"legal_entity_id": "buyer_id", TARGET_ID_COL: "predicted_id"})
    )
    covered = cold_sub["buyer_id"].nunique() if not cold_sub.empty else 0
    print(f"Cold core predictions: {len(cold_sub):,} across {covered}/{len(cold_ids)} buyers")


# 6) Submission
submission = (
    pd.concat([warm_sub, cold_sub], ignore_index=True)
    .drop_duplicates()
    .sort_values(["buyer_id", "predicted_id"])
    .reset_index(drop=True)
)
submission.to_csv(OUT / "submission_level3.csv", index=False)
print(f"Saved submission_level3.csv with {len(submission):,} rows")
