"""
Simple Two-Model Core Demand Pipeline (Rolling Validation)
===========================================================
Model A: Binary classifier -> predicts if a buyer will purchase an eclass (yes/no)
Model B: Regressor -> predicts how much they will spend if they purchase

Expected spend = P(purchase) * PredictedSpendIfPurchase
Expected savings = Expected spend * SAVINGS_RATE
Core demand decision: Expected savings > one-time fee (FEE)

Enhancements:
  1) Rolling time validation over two windows
  2) Hyperparameter tuning directly on net Euro score
  3) Warm and cold-proxy metrics tracked separately

Output:
  ../submission.csv
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
LABEL_MONTHS = 6  # kept for reference
TOTAL_FEE = FEE   # fee is applied once per predicted element

FEATURE_CUTOFF = pd.Timestamp("2024-12-31")
LABEL_END = pd.Timestamp("2025-06-30")

# Requested rolling windows
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

# Initial values, overwritten by tuning
EXPECTED_SPEND_SCALE = 1.1
COLD_EXPECTED_DISCOUNT = 0.9
COLD_TOP_K = 200

# Small search space for direct net-score optimization
SCALE_GRID = [1.0, 1.2, 1.4]
COLD_DISCOUNT_GRID = [0.7, 0.8, 0.9, 0.95]
COLD_TOPK_GRID = [185, 200, 215, 230]

LGBM_PARAMS_CLF = dict(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

LGBM_PARAMS_REG = dict(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "unite"
OUT = Path(__file__).resolve().parent.parent
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_pair_features(df: pd.DataFrame, cutoff: pd.Timestamp, buyer_info: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (buyer, eclass)."""
    grp = df.groupby(["legal_entity_id", "eclass"])

    feats = pd.concat([
        grp["line_value"].sum().rename("total_spend"),
        grp.size().rename("n_orders"),
        grp["ym"].nunique().rename("n_months_active"),
        grp["orderdate"].max().rename("last_order"),
        grp["orderdate"].min().rename("first_order"),
    ], axis=1).reset_index()

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

    buyer_total = (
        df.groupby("legal_entity_id")["line_value"].sum().rename("buyer_total_spend").reset_index()
    )
    eclass_total = (
        df.groupby("eclass")["line_value"].sum().rename("eclass_total_spend").reset_index()
    )
    feats = feats.merge(buyer_total, on="legal_entity_id", how="left")
    feats = feats.merge(eclass_total, on="eclass", how="left")

    feats = feats.merge(
        buyer_info[[
            "legal_entity_id", "log_employees", "section_enc", "nace_2digits", "has_secondary_nace"
        ]],
        on="legal_entity_id",
        how="left",
    )

    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        feats[c] = feats[c].fillna(-1)

    return feats.drop(columns=["last_order", "first_order"])


def infer_expected_spend(
    feats: pd.DataFrame,
    clf,
    reg,
    feature_cols: list[str],
    expected_spend_scale: float,
) -> pd.DataFrame:
    """Run two models and compute expected spend/savings/core decision."""
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


def train_two_models(train_feats: pd.DataFrame, feature_cols: list[str]):
    """Train the classifier and regressor."""
    X = train_feats[feature_cols].fillna(0)

    clf = lgb.LGBMClassifier(**LGBM_PARAMS_CLF)
    clf.fit(X, train_feats["buy_label"])

    pos_mask = train_feats["future_spend"] > 0
    reg = lgb.LGBMRegressor(**LGBM_PARAMS_REG)
    reg.fit(X.loc[pos_mask], train_feats.loc[pos_mask, "log_spend_label"])

    return clf, reg


def save_feature_importance_plots(clf, reg, feature_cols: list[str], out_dir: Path) -> None:
    """Save feature-importance visualizations for both models and a combined view."""
    clf_imp = pd.Series(clf.feature_importances_, index=feature_cols, name="clf_importance")
    reg_imp = pd.Series(reg.feature_importances_, index=feature_cols, name="reg_importance")

    imp_df = pd.concat([clf_imp, reg_imp], axis=1).fillna(0)

    # Normalize so both models contribute on same scale in combined view.
    imp_df["clf_norm"] = imp_df["clf_importance"] / max(imp_df["clf_importance"].sum(), 1)
    imp_df["reg_norm"] = imp_df["reg_importance"] / max(imp_df["reg_importance"].sum(), 1)
    imp_df["combined_importance"] = 0.5 * imp_df["clf_norm"] + 0.5 * imp_df["reg_norm"]

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    imp_df.sort_values("clf_importance", ascending=True)["clf_importance"].plot.barh(
        ax=axes[0], color="#3b82f6", edgecolor="white"
    )
    axes[0].set_title("Classifier Importance\nP(purchase)")
    axes[0].set_xlabel("Importance (split count)")

    imp_df.sort_values("reg_importance", ascending=True)["reg_importance"].plot.barh(
        ax=axes[1], color="#10b981", edgecolor="white"
    )
    axes[1].set_title("Regressor Importance\nSpend if purchased")
    axes[1].set_xlabel("Importance (split count)")

    imp_df.sort_values("combined_importance", ascending=True)["combined_importance"].plot.barh(
        ax=axes[2], color="#f59e0b", edgecolor="white"
    )
    axes[2].set_title("Combined Predictive Importance")
    axes[2].set_xlabel("Normalized importance")

    plt.suptitle("Main Predictor Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_file = out_dir / "main_predictor_features.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"Saved feature visualization: {out_file}")
    print("Top 10 combined predictors:")
    print(
        imp_df.sort_values("combined_importance", ascending=False)
        .head(10)[["clf_importance", "reg_importance", "combined_importance"]]
        .round(4)
        .to_string()
    )


def realized_net(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, int, int, int]:
    """Compute realized net score against known future spend."""
    merged = pred_df[["legal_entity_id", "eclass", "is_core"]].merge(
        truth_df, on=["legal_entity_id", "eclass"], how="left"
    )
    merged["future_spend"] = merged["future_spend"].fillna(0)

    earned = (merged["is_core"] * merged["future_spend"] * SAVINGS_RATE).sum()
    fees = merged["is_core"].sum() * TOTAL_FEE
    net = float(earned - fees)

    profitable = (merged["future_spend"] * SAVINGS_RATE > TOTAL_FEE).astype(int)
    tp = int(((merged["is_core"] == 1) & (profitable == 1)).sum())
    fp = int(((merged["is_core"] == 1) & (profitable == 0)).sum())
    return net, int(merged["is_core"].sum()), tp, fp


def build_cold_candidates_from_warm(
    warm_pred: pd.DataFrame,
    buyer_info: pd.DataFrame,
    cold_ids: set,
    feature_cols: list[str],
    cold_top_k: int,
) -> pd.DataFrame:
    """Create cold candidate rows by sector-level imputation from warm predictions."""
    impute_cols = feature_cols.copy()
    for c in ["log_employees", "section_enc", "nace_2digits", "has_secondary_nace"]:
        impute_cols.remove(c)

    sector_stats = (
        warm_pred.groupby(["nace_2digits", "eclass"])
        .agg(
            **{f"mean_{c}": (c, "mean") for c in impute_cols},
            n_buyers=("legal_entity_id", "nunique"),
            avg_expected_spend=("expected_spend", "mean"),
        )
        .reset_index()
    )

    global_stats = (
        warm_pred.groupby("eclass")
        .agg(
            **{f"mean_{c}": (c, "mean") for c in impute_cols},
            n_buyers=("legal_entity_id", "nunique"),
            avg_expected_spend=("expected_spend", "mean"),
        )
        .reset_index()
    )

    cold_rows = []
    for _, b in buyer_info[buyer_info["legal_entity_id"].isin(cold_ids)].iterrows():
        bid = b["legal_entity_id"]
        n2 = b["nace_2digits"]

        cands = sector_stats[sector_stats["nace_2digits"] == n2]
        if cands.empty:
            cands = global_stats.copy()

        cands = cands.sort_values(["n_buyers", "avg_expected_spend"], ascending=False).head(cold_top_k)

        for _, r in cands.iterrows():
            row = {
                "legal_entity_id": bid,
                "eclass": r["eclass"],
                "log_employees": b["log_employees"],
                "section_enc": b["section_enc"],
                "nace_2digits": b["nace_2digits"],
                "has_secondary_nace": b["has_secondary_nace"],
            }
            for c in impute_cols:
                row[c] = r.get(f"mean_{c}", 0)
            cold_rows.append(row)

    return pd.DataFrame(cold_rows)


def evaluate_fold(
    plis: pd.DataFrame,
    buyer_info: pd.DataFrame,
    warm_train_ids: set,
    warm_eval_ids: set,
    pseudo_cold_ids: set,
    feature_cols: list[str],
    fold: dict,
    expected_spend_scale: float,
    cold_expected_discount: float,
    cold_top_k: int,
) -> dict:
    """Train on train IDs; evaluate warm and cold proxy separately."""
    train_end = fold["train_end"]
    val_start = fold["val_start"]
    val_end = fold["val_end"]

    feat_df = plis[(plis["orderdate"] <= train_end) & (plis["legal_entity_id"].isin(warm_train_ids))].copy()
    label_df = plis[
        (plis["orderdate"] >= val_start)
        & (plis["orderdate"] <= val_end)
        & (plis["legal_entity_id"].isin(warm_train_ids.union(pseudo_cold_ids)))
    ].copy()

    train_feats = build_pair_features(feat_df, train_end, buyer_info)
    future_spend_train = (
        label_df[label_df["legal_entity_id"].isin(warm_train_ids)]
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    train_feats = train_feats.merge(future_spend_train, on=["legal_entity_id", "eclass"], how="left")
    train_feats["future_spend"] = train_feats["future_spend"].fillna(0)
    train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
    train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

    if train_feats.empty:
        return {
            "fold": fold["name"],
            "warm_net": -np.inf,
            "cold_proxy_net": -np.inf,
            "combined_net": -np.inf,
        }

    clf, reg = train_two_models(train_feats, feature_cols)

    # Warm evaluation
    warm_eval_feat_df = plis[
        (plis["orderdate"] <= train_end)
        & (plis["legal_entity_id"].isin(warm_eval_ids))
    ].copy()
    warm_eval_feats = build_pair_features(warm_eval_feat_df, train_end, buyer_info)
    warm_eval_pred = infer_expected_spend(
        warm_eval_feats, clf, reg, feature_cols, expected_spend_scale
    )

    warm_truth = (
        label_df[label_df["legal_entity_id"].isin(warm_eval_ids)]
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_eval_pred, warm_truth)

    # Cold proxy evaluation
    source_warm_pred = infer_expected_spend(
        train_feats[["legal_entity_id", "eclass"] + feature_cols],
        clf,
        reg,
        feature_cols,
        expected_spend_scale,
    )

    cold_candidates = build_cold_candidates_from_warm(
        source_warm_pred,
        buyer_info,
        pseudo_cold_ids,
        feature_cols,
        cold_top_k,
    )

    if cold_candidates.empty:
        cold_net, cold_n, cold_tp, cold_fp = 0.0, 0, 0, 0
    else:
        cold_pred = infer_expected_spend(
            cold_candidates, clf, reg, feature_cols, expected_spend_scale
        )
        cold_pred["expected_spend"] *= cold_expected_discount
        cold_pred["expected_savings"] = cold_pred["expected_spend"] * SAVINGS_RATE
        cold_pred["is_core"] = (cold_pred["expected_savings"] > TOTAL_FEE).astype(int)

        cold_truth = (
            label_df[label_df["legal_entity_id"].isin(pseudo_cold_ids)]
            .groupby(["legal_entity_id", "eclass"])["line_value"]
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

FEATURE_COLS = [
    "total_spend", "n_orders", "n_months_active", "recency_months", "span_months",
    "monthly_avg", "last6m_spend", "buyer_total_spend", "eclass_total_spend",
    "log_employees", "section_enc", "nace_2digits", "has_secondary_nace",
]


# -----------------------------------------------------------------------------
# 2) Rolling validation and direct net-score tuning
# -----------------------------------------------------------------------------
print("\nRunning rolling validation + net-score tuning...")

# Build a pseudo-cold set from warm buyers for separate cold proxy tracking.
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
for scale, cdisc, ctopk in product(SCALE_GRID, COLD_DISCOUNT_GRID, COLD_TOPK_GRID):
    fold_metrics = []
    for fold in ROLLING_FOLDS:
        fold_metrics.append(
            evaluate_fold(
                plis=plis,
                buyer_info=buyer_info,
                warm_train_ids=warm_train_ids,
                warm_eval_ids=warm_eval_ids,
                pseudo_cold_ids=pseudo_cold_ids,
                feature_cols=FEATURE_COLS,
                fold=fold,
                expected_spend_scale=scale,
                cold_expected_discount=cdisc,
                cold_top_k=ctopk,
            )
        )

    fold_df = pd.DataFrame(fold_metrics)
    search_rows.append({
        "scale": scale,
        "cold_discount": cdisc,
        "cold_top_k": ctopk,
        "avg_warm_net": float(fold_df["warm_net"].mean()),
        "avg_cold_proxy_net": float(fold_df["cold_proxy_net"].mean()),
        "avg_combined_net": float(fold_df["combined_net"].mean()),
    })

search_df = pd.DataFrame(search_rows).sort_values("avg_combined_net", ascending=False).reset_index(drop=True)
print("\nTop tuning configs (objective = avg_combined_net):")
print(search_df.head(10).to_string(index=False))

best = search_df.iloc[0]
EXPECTED_SPEND_SCALE = float(best["scale"])
COLD_EXPECTED_DISCOUNT = float(best["cold_discount"])
COLD_TOP_K = int(best["cold_top_k"])

print("\nChosen params:")
print(f"  EXPECTED_SPEND_SCALE   = {EXPECTED_SPEND_SCALE}")
print(f"  COLD_EXPECTED_DISCOUNT = {COLD_EXPECTED_DISCOUNT}")
print(f"  COLD_TOP_K             = {COLD_TOP_K}")
print(f"  Avg warm net           = EUR {best['avg_warm_net']:,.2f}")
print(f"  Avg cold proxy net     = EUR {best['avg_cold_proxy_net']:,.2f}")
print(f"  Avg combined net       = EUR {best['avg_combined_net']:,.2f}")


# -----------------------------------------------------------------------------
# 3) Final training on canonical split
# -----------------------------------------------------------------------------
print("\nTraining final models on canonical split...")
feat_df = plis[(plis["orderdate"] <= FEATURE_CUTOFF) & (plis["legal_entity_id"].isin(warm_ids))].copy()
label_df = plis[
    (plis["orderdate"] > FEATURE_CUTOFF)
    & (plis["orderdate"] <= LABEL_END)
    & (plis["legal_entity_id"].isin(warm_ids))
].copy()

train_feats = build_pair_features(feat_df, FEATURE_CUTOFF, buyer_info)
future_spend = (
    label_df.groupby(["legal_entity_id", "eclass"])["line_value"]
    .sum()
    .rename("future_spend")
    .reset_index()
)
train_feats = train_feats.merge(future_spend, on=["legal_entity_id", "eclass"], how="left")
train_feats["future_spend"] = train_feats["future_spend"].fillna(0)
train_feats["buy_label"] = (train_feats["future_spend"] > 0).astype(int)
train_feats["log_spend_label"] = np.log1p(train_feats["future_spend"])

clf, reg = train_two_models(train_feats, FEATURE_COLS)
save_feature_importance_plots(clf, reg, FEATURE_COLS, PLOTS)

warm_holdout_pred = infer_expected_spend(
    train_feats[["legal_entity_id", "eclass"] + FEATURE_COLS],
    clf,
    reg,
    FEATURE_COLS,
    EXPECTED_SPEND_SCALE,
)
warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_holdout_pred, future_spend)
print("\nCanonical warm holdout metrics:")
print(f"  warm_net = EUR {warm_net:,.2f} | preds={warm_n} | tp={warm_tp} | fp={warm_fp}")


# -----------------------------------------------------------------------------
# 4) Final predictions: warm
# -----------------------------------------------------------------------------
pred_cutoff = plis["orderdate"].max()
warm_full = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
warm_pred = build_pair_features(warm_full, pred_cutoff, buyer_info)
warm_pred = infer_expected_spend(
    warm_pred,
    clf,
    reg,
    FEATURE_COLS,
    EXPECTED_SPEND_SCALE,
)

warm_sub = (
    warm_pred.loc[warm_pred["is_core"] == 1, ["legal_entity_id", "eclass"]]
    .rename(columns={"legal_entity_id": "buyer_id", "eclass": "cluster"})
)
print(f"Warm core predictions: {len(warm_sub):,}")


# -----------------------------------------------------------------------------
# 5) Final predictions: cold
# -----------------------------------------------------------------------------
cold_candidates = build_cold_candidates_from_warm(
    warm_pred,
    buyer_info,
    cold_ids,
    FEATURE_COLS,
    COLD_TOP_K,
)

if cold_candidates.empty:
    cold_sub = pd.DataFrame(columns=["buyer_id", "cluster"])
    print("Cold: no candidates generated")
else:
    cold_pred = infer_expected_spend(
        cold_candidates,
        clf,
        reg,
        FEATURE_COLS,
        EXPECTED_SPEND_SCALE,
    )
    cold_pred["expected_spend"] = cold_pred["expected_spend"] * COLD_EXPECTED_DISCOUNT
    cold_pred["expected_savings"] = cold_pred["expected_spend"] * SAVINGS_RATE
    cold_pred["is_core"] = (cold_pred["expected_savings"] > TOTAL_FEE).astype(int)

    cold_sub = (
        cold_pred.loc[cold_pred["is_core"] == 1, ["legal_entity_id", "eclass"]]
        .rename(columns={"legal_entity_id": "buyer_id", "eclass": "cluster"})
    )
    covered = cold_sub["buyer_id"].nunique() if not cold_sub.empty else 0
    print(f"Cold core predictions: {len(cold_sub):,} across {covered}/{len(cold_ids)} buyers")


# -----------------------------------------------------------------------------
# 6) Submission
# -----------------------------------------------------------------------------
submission = (
    pd.concat([warm_sub], ignore_index=True)
    .sort_values(["buyer_id", "cluster"])
    .reset_index(drop=True)
)

submission.to_csv(OUT / "submission.csv", index=False)
print(f"Saved submission.csv with {len(submission):,} rows ({len(warm_sub)} warm + {len(cold_sub)} cold)")
