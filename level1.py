"""
predict_core_demand.py
----------------------
Binary classifier: for each (warm-start buyer, eclass) pair, predict whether
projected savings exceed the recurring fee.

  Label = 1  if  (future_spend * SAVINGS_RATE)  >  (FEE * LABEL_MONTHS)

Strategy
--------
- Feature window : Jan 2023 – Dec 2024  (used to compute per-pair signals)
- Label window   : Jan 2025 – Jun 2025  (pseudo ground-truth held out for training)
- Prediction     : full history Jan 2023 – Jun 2025 used as features for test buyers

Tunable parameters
------------------
  SAVINGS_RATE  – fraction of spend captured as savings (default 0.10 = 10 %)
  FEE           – fixed monthly fee per predicted core demand eclass in € (default 10.0)
  LABEL_MONTHS  – number of months in the label window (default 6)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# ── Tunable parameters ────────────────────────────────────────────────────────
SAVINGS_RATE  = 0.10   # 10 % savings on matched future spend
FEE           = 10.0   # € fixed monthly fee per core demand eclass
LABEL_MONTHS  = 6      # months in the pseudo label window

# Classification threshold: predict core if prob >= THRESHOLD.
# 0.5 = default sklearn behaviour. Lower = higher recall, more FPs.
# Set to None to auto-select the threshold that maximises holdout net score.
THRESHOLD = None

# Set to a list of legal_entity_ids to run on a subset only (fast debug).
# Set to None to run on all buyers.
DEBUG_BUYER_IDS = None  # e.g. [61457883, 12345678]

# ── Model hyperparameters ─────────────────────────────────────────────────────
N_ESTIMATORS  = 200    # number of boosting trees
MAX_DEPTH     = 4      # max depth per tree
LEARNING_RATE = 0.05   # shrinkage
SUBSAMPLE     = 0.8    # fraction of samples per tree (stochastic boosting)

# ── Date boundaries ───────────────────────────────────────────────────────────
FEATURE_CUTOFF = pd.Timestamp("2024-12-31")  # features use data up to here
LABEL_END      = pd.Timestamp("2025-06-30")  # label window is (FEATURE_CUTOFF, LABEL_END]

DATA = Path("data/unite")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading customer_test.csv ...")
customers = pd.read_csv(DATA / "customer_test.csv", sep="\t")

print("Loading nace_codes.csv ...")
nace = pd.read_csv(DATA / "nace_codes.csv", sep="\t")
nace["nace_code"] = nace["nace_code"].astype(int)

# Encode toplevel_section as integer (A=0, B=1, ...)
sections = sorted(nace["toplevel_section"].dropna().unique())
section_map = {s: i for i, s in enumerate(sections)}

# Build a buyer-level info table with nace hierarchy + company signals
customers["nace_code"] = customers["nace_code"].astype(int)
buyer_info = customers.merge(
    nace[["nace_code", "toplevel_section", "nace_2digits"]],
    on="nace_code", how="left"
)
buyer_info["toplevel_section_enc"] = (
    buyer_info["toplevel_section"].map(section_map).fillna(-1).astype(int)
)
buyer_info["nace_2digits"]       = buyer_info["nace_2digits"].fillna(-1)
buyer_info["has_secondary_nace"] = buyer_info["secondary_nace_code"].notna().astype(int)
buyer_info["log_employees"]      = np.log1p(buyer_info["estimated_number_employees"].fillna(0))

print("Loading plis_training.csv (chunked) ...")
chunks = []
for chunk in pd.read_csv(
    DATA / "plis_training.csv", sep="\t", low_memory=False,
    usecols=["orderdate", "legal_entity_id", "eclass",
             "quantityvalue", "vk_per_item"],
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

print(f"  {len(plis):,} rows  |  "
      f"{plis['orderdate'].min().date()} → {plis['orderdate'].max().date()}")

# Always restrict to the 100 test buyers — they are the only ones we need
test_buyer_ids = set(customers["legal_entity_id"])
plis = plis[plis["legal_entity_id"].isin(test_buyer_ids)].copy()
print(f"  Filtered to {len(test_buyer_ids)} test buyers → {len(plis):,} rows")

# Additional debug filter to a specific subset (fast dev runs)
if DEBUG_BUYER_IDS is not None:
    plis = plis[plis["legal_entity_id"].isin(DEBUG_BUYER_IDS)].copy()
    print(f"  [DEBUG] Further filtered to {len(DEBUG_BUYER_IDS)} buyer(s): {DEBUG_BUYER_IDS}  → {len(plis):,} rows")


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, cutoff: pd.Timestamp, binfo: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (legal_entity_id, eclass) with predictive signals."""
    grp = df.groupby(["legal_entity_id", "eclass"])

    feats = pd.concat([
        grp["line_value"].sum().rename("total_spend"),
        grp.size().rename("n_orders"),
        grp["ym"].nunique().rename("n_months_active"),
        grp["orderdate"].max().rename("last_order"),
        grp["orderdate"].min().rename("first_order"),
    ], axis=1).reset_index()

    feats["recency_months"]  = (cutoff - feats["last_order"]).dt.days / 30
    feats["span_months"]     = ((feats["last_order"] - feats["first_order"]).dt.days / 30).clip(lower=1)
    feats["monthly_avg"]     = feats["total_spend"] / feats["span_months"]
    feats["activity_ratio"]  = feats["n_months_active"] / feats["span_months"].clip(lower=1)
    # Coefficient of variation of monthly spend (low CV = stable = good)
    monthly = (df.groupby(["legal_entity_id", "eclass", "ym"])["line_value"]
                 .sum().reset_index())
    cv_grp  = monthly.groupby(["legal_entity_id", "eclass"])["line_value"]
    cv_std  = cv_grp.std().fillna(0)
    cv_mean = cv_grp.mean()
    cv = (cv_std / cv_mean.where(cv_mean > 0, other=1)).rename("spend_cv")
    feats = feats.merge(cv.reset_index(), on=["legal_entity_id", "eclass"], how="left")
    feats["spend_cv"].fillna(0, inplace=True)

    # Spend in the last 6 months of the feature window (recency signal)
    last6_start = cutoff - pd.DateOffset(months=6)
    last6 = (df[df["orderdate"] > last6_start]
               .groupby(["legal_entity_id", "eclass"])["line_value"]
               .sum().rename("last6m_spend"))
    feats = feats.merge(last6.reset_index(), on=["legal_entity_id", "eclass"], how="left")
    feats["last6m_spend"].fillna(0, inplace=True)

    # Buyer-level total spend (size proxy)
    buyer_total = (df.groupby("legal_entity_id")["line_value"]
                     .sum().rename("buyer_total_spend").reset_index())
    feats = feats.merge(buyer_total, on="legal_entity_id", how="left")

    # Eclass popularity across all buyers (category signal)
    eclass_total = (df.groupby("eclass")["line_value"]
                      .sum().rename("eclass_total_spend").reset_index())
    feats = feats.merge(eclass_total, on="eclass", how="left")

    # Merge buyer-level features: NACE hierarchy + company size
    buyer_cols = ["legal_entity_id", "log_employees",
                  "toplevel_section_enc", "nace_code", "secondary_nace_code"]
    feats = feats.merge(binfo[buyer_cols], on="legal_entity_id", how="left")
    feats["toplevel_section_enc"] = feats["toplevel_section_enc"].fillna(-1).astype(int)
    feats["nace_code"]          = feats["nace_code"].fillna(-1)
    feats["secondary_nace_code"]    = feats["secondary_nace_code"].fillna(0).astype(int)
    feats["log_employees"]         = feats["log_employees"].fillna(0)

    feats.drop(columns=["last_order", "first_order"], inplace=True)
    return feats


FEATURE_COLS = [
    "n_orders", "n_months_active",
    "recency_months", "span_months", "monthly_avg",
    "activity_ratio", "log_employees", "eclass_total_spend", "nace_code", "secondary_nace_code",
]


# ── Build training dataset ────────────────────────────────────────────────────
feat_df  = plis[plis["orderdate"] <= FEATURE_CUTOFF].copy()
label_df = plis[(plis["orderdate"] > FEATURE_CUTOFF) &
                (plis["orderdate"] <= LABEL_END)].copy()

print("Building training features (Jan 2023 – Dec 2024) ...")
train_feats = build_features(feat_df, FEATURE_CUTOFF, buyer_info)

# Attach label: was savings > total fee over the label window?
future_spend = (label_df.groupby(["legal_entity_id", "eclass"])["line_value"]
                         .sum().rename("future_spend").reset_index())
train_feats  = train_feats.merge(future_spend, on=["legal_entity_id", "eclass"], how="left")
train_feats["future_spend"].fillna(0, inplace=True)

total_fee  = FEE * LABEL_MONTHS
train_feats["savings"] = train_feats["future_spend"] * SAVINGS_RATE
train_feats["label"]   = (train_feats["savings"] > total_fee).astype(int)

pos_rate = train_feats["label"].mean()
print(f"  {len(train_feats):,} training pairs  |  "
      f"positive (core) rate: {pos_rate:.1%}  "
      f"(break-even spend per 6 months: €{total_fee / SAVINGS_RATE:,.0f})")


# ── Train classifier ──────────────────────────────────────────────────────────
X = train_feats[FEATURE_COLS].fillna(0)
y = train_feats["label"]

print("Training GradientBoostingClassifier ...")
clf = GradientBoostingClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    subsample=SUBSAMPLE,
    random_state=42,
)
clf.fit(X, y)

print("\nIn-sample classification report:")
print(classification_report(y, clf.predict(X), target_names=["not-core", "core"]))

print("Feature importances:")
fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(fi.to_string())


# ── Holdout evaluation on Jan–Jun 2025 (visible test window) ────────────────
# Use features from Jan 2023 – Dec 2024 (same as training) and evaluate
# how well the trained model predicts the Jan–Jun 2025 period.
print("\nHoldout evaluation (Jan–Jun 2025) ...")
test_pairs = label_df.groupby(["legal_entity_id", "eclass"])["line_value"].sum().rename("holdout_spend").reset_index()
test_pairs["holdout_savings"] = test_pairs["holdout_spend"] * SAVINGS_RATE
test_pairs["true_label"]      = (test_pairs["holdout_savings"] > FEE * LABEL_MONTHS).astype(int)

# Attach model features (computed on feature window)
holdout_eval = train_feats[["legal_entity_id", "eclass"] + FEATURE_COLS].merge(
    test_pairs[["legal_entity_id", "eclass", "true_label", "holdout_spend"]],
    on=["legal_entity_id", "eclass"], how="inner"
)

X_holdout = holdout_eval[FEATURE_COLS].fillna(0)
holdout_eval["pred_prob"] = clf.predict_proba(X_holdout)[:, 1]

# ── Threshold sweep: find the threshold that maximises holdout net score ──────
thresholds = np.arange(0.05, 0.96, 0.05)
sweep = []
for t in thresholds:
    pred = (holdout_eval["pred_prob"] >= t).astype(int)
    earned = pred * holdout_eval["holdout_spend"] * SAVINGS_RATE
    fees   = pred * FEE * LABEL_MONTHS
    net    = (earned - fees).sum()
    tp = ((pred == 1) & (holdout_eval["true_label"] == 1)).sum()
    fp = ((pred == 1) & (holdout_eval["true_label"] == 0)).sum()
    sweep.append({"threshold": round(t, 2), "net_score": round(net, 2),
                  "predicted": int(pred.sum()), "tp": int(tp), "fp": int(fp)})

sweep_df = pd.DataFrame(sweep)
print("\nThreshold sweep (holdout Jan–Jun 2025):")
print(sweep_df.to_string(index=False))

best_t = sweep_df.loc[sweep_df["net_score"].idxmax(), "threshold"]
if THRESHOLD is None:
    chosen_t = best_t
    print(f"\nAuto-selected threshold: {chosen_t} (best net score on holdout)")
else:
    chosen_t = THRESHOLD
    print(f"\nUsing fixed threshold: {chosen_t}  (best on holdout was {best_t})")

holdout_eval["pred_label"] = (holdout_eval["pred_prob"] >= chosen_t).astype(int)

# Economic score on holdout at chosen threshold
holdout_eval["earned_savings"] = holdout_eval["pred_label"] * holdout_eval["holdout_spend"] * SAVINGS_RATE
holdout_eval["paid_fees"]      = holdout_eval["pred_label"] * FEE * LABEL_MONTHS
holdout_eval["net_benefit"]    = holdout_eval["earned_savings"] - holdout_eval["paid_fees"]

holdout_score = holdout_eval["net_benefit"].sum()
print(f"\nHoldout results at threshold={chosen_t}:")
print(f"  Pairs in holdout           : {len(holdout_eval):,}")
print(f"  Predicted core             : {holdout_eval['pred_label'].sum()}")
print(f"  True positives             : {((holdout_eval['pred_label']==1) & (holdout_eval['true_label']==1)).sum()}")
print(f"  False positives (fee waste): {((holdout_eval['pred_label']==1) & (holdout_eval['true_label']==0)).sum()}")
print(f"  Net economic score (€)     : {holdout_score:,.2f}")
print()
print("Holdout classification report:")
print(classification_report(
    holdout_eval["true_label"], holdout_eval["pred_label"],
    target_names=["not-core", "core"], zero_division=0
))


# ── Predict for test buyers (warm-start only) ─────────────────────────────────
print("Building prediction features (full history Jan 2023 – Jun 2025) ...")
PRED_CUTOFF = plis["orderdate"].max()
warm_ids    = set(customers.loc[customers["task"] == "predict future", "legal_entity_id"])
if DEBUG_BUYER_IDS is not None:
    warm_ids = warm_ids & set(DEBUG_BUYER_IDS)
pred_df     = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
pred_feats  = build_features(pred_df, PRED_CUTOFF, buyer_info)

X_pred = pred_feats[FEATURE_COLS].fillna(0)
pred_feats["core_prob"] = clf.predict_proba(X_pred)[:, 1]
pred_feats["is_core"]   = (pred_feats["core_prob"] >= chosen_t).astype(int)

n_core   = pred_feats["is_core"].sum()
n_buyers = pred_feats.loc[pred_feats["is_core"] == 1, "legal_entity_id"].nunique()
print(f"  {len(pred_feats):,} pairs evaluated  |  {n_core} predicted core across {n_buyers} buyers")


# ── Assemble and save submission ──────────────────────────────────────────────
submission = (
    pred_feats[pred_feats["is_core"] == 1][["legal_entity_id", "eclass"]]
    .rename(columns={"legal_entity_id": "buyer_id", "eclass": "cluster"})
    .sort_values(["buyer_id", "cluster"])
    .reset_index(drop=True)
)

submission.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv  ({len(submission):,} rows)")
print(submission.head(10).to_string(index=False))
