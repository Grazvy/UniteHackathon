"""Entrypoint for v3 modular two-model pipeline.

Run:
  /Users/sarpsahinalp/projects/UniteHackathon/.venv/bin/python v3/run_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from v3.config import (
    SAVINGS_RATE,
    TOTAL_FEE,
    FEATURE_CUTOFF,
    LABEL_END,
    ROLLING_FOLDS,
    SCALE_GRID,
    COLD_DISCOUNT_GRID,
    COLD_TOPK_GRID,
    DATA_DIR,
    OUT_DIR,
    PLOTS_DIR,
)
from v3.data.loader import load_core_data
from v3.features.schema import FEATURE_COLS
from v3.features.pair_features import build_pair_features
from v3.models.two_model import train_two_models, infer_expected_spend
from v3.evaluation.scoring import realized_net
from v3.evaluation.rolling import tune_parameters
from v3.cold_start.candidates import build_cold_candidates_from_warm
from v3.viz.feature_importance import save_feature_importance_plots
from v3.io.submission import build_submission_df


def main():
    print("Loading data...")
    customers, nace, buyer_info, plis, warm_ids, cold_ids = load_core_data(DATA_DIR)
    print(f"Rows: {len(plis):,} | Warm buyers: {len(warm_ids)} | Cold buyers: {len(cold_ids)}")

    print("\nRunning rolling validation + net-score tuning...")
    tuning = tune_parameters(
        plis=plis,
        buyer_info=buyer_info,
        warm_ids=warm_ids,
        feature_cols=FEATURE_COLS,
        rolling_folds=ROLLING_FOLDS,
        scale_grid=SCALE_GRID,
        cold_discount_grid=COLD_DISCOUNT_GRID,
        cold_topk_grid=COLD_TOPK_GRID,
        savings_rate=SAVINGS_RATE,
        total_fee=TOTAL_FEE,
        random_state=42,
    )

    best_scale = tuning["best_scale"]
    best_cold_discount = tuning["best_cold_discount"]
    best_cold_top_k = tuning["best_cold_top_k"]
    search_df = tuning["search_df"]

    print("Top tuning configs (objective = avg_combined_net):")
    print(search_df.head(10).to_string(index=False))
    print("\nChosen params:")
    print(f"  EXPECTED_SPEND_SCALE   = {best_scale}")
    print(f"  COLD_EXPECTED_DISCOUNT = {best_cold_discount}")
    print(f"  COLD_TOP_K             = {best_cold_top_k}")

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

    plot_path, imp_df = save_feature_importance_plots(clf, reg, FEATURE_COLS, PLOTS_DIR)
    print(f"Saved feature importance plot: {plot_path}")

    warm_holdout_pred = infer_expected_spend(
        train_feats[["legal_entity_id", "eclass"] + FEATURE_COLS],
        clf,
        reg,
        FEATURE_COLS,
        best_scale,
    )
    warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_holdout_pred, future_spend)
    print("\nCanonical warm holdout metrics:")
    print(f"  warm_net = EUR {warm_net:,.2f} | preds={warm_n} | tp={warm_tp} | fp={warm_fp}")

    print("\nFinal predictions: warm")
    pred_cutoff = plis["orderdate"].max()
    warm_full = plis[plis["legal_entity_id"].isin(warm_ids)].copy()
    warm_pred = build_pair_features(warm_full, pred_cutoff, buyer_info)
    warm_pred = infer_expected_spend(warm_pred, clf, reg, FEATURE_COLS, best_scale)

    print("Final predictions: cold")
    cold_candidates = build_cold_candidates_from_warm(
        warm_pred,
        buyer_info,
        cold_ids,
        FEATURE_COLS,
        best_cold_top_k,
    )

    if cold_candidates.empty:
        cold_pred = None
        print("  No cold candidates generated")
    else:
        cold_pred = infer_expected_spend(cold_candidates, clf, reg, FEATURE_COLS, best_scale)
        cold_pred["expected_spend"] *= best_cold_discount
        cold_pred["expected_savings"] = cold_pred["expected_spend"] * SAVINGS_RATE
        cold_pred["is_core"] = (cold_pred["expected_savings"] > TOTAL_FEE).astype(int)
        covered = cold_pred.loc[cold_pred["is_core"] == 1, "legal_entity_id"].nunique()
        print(f"  Cold core predictions: {int(cold_pred['is_core'].sum()):,} across {covered}/{len(cold_ids)} buyers")

    submission, warm_sub, cold_sub = build_submission_df(warm_pred, cold_pred)
    out_file = OUT_DIR / "submission.csv"
    submission.to_csv(out_file, index=False)

    print(f"\nSaved submission: {out_file}")
    print(f"Rows: {len(submission):,} ({len(warm_sub)} warm + {len(cold_sub)} cold)")


if __name__ == "__main__":
    main()
