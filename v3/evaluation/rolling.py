import numpy as np
import pandas as pd
from itertools import product

from v3.features.pair_features import build_pair_features
from v3.models.two_model import train_two_models, infer_expected_spend
from v3.cold_start.candidates import build_cold_candidates_from_warm
from v3.evaluation.scoring import realized_net


def evaluate_fold(
    plis,
    buyer_info,
    warm_train_ids,
    warm_eval_ids,
    pseudo_cold_ids,
    feature_cols,
    fold,
    expected_spend_scale,
    cold_expected_discount,
    cold_top_k,
    savings_rate,
    total_fee,
):
    """Train on historical window and evaluate warm + cold-proxy net in the validation window."""
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

    warm_eval_feat_df = plis[
        (plis["orderdate"] <= train_end)
        & (plis["legal_entity_id"].isin(warm_eval_ids))
    ].copy()
    warm_eval_feats = build_pair_features(warm_eval_feat_df, train_end, buyer_info)
    warm_eval_pred = infer_expected_spend(warm_eval_feats, clf, reg, feature_cols, expected_spend_scale)

    warm_truth = (
        label_df[label_df["legal_entity_id"].isin(warm_eval_ids)]
        .groupby(["legal_entity_id", "eclass"])["line_value"]
        .sum()
        .rename("future_spend")
        .reset_index()
    )
    warm_net, warm_n, warm_tp, warm_fp = realized_net(warm_eval_pred, warm_truth)

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
        cold_pred = infer_expected_spend(cold_candidates, clf, reg, feature_cols, expected_spend_scale)
        cold_pred["expected_spend"] *= cold_expected_discount
        cold_pred["expected_savings"] = cold_pred["expected_spend"] * savings_rate
        cold_pred["is_core"] = (cold_pred["expected_savings"] > total_fee).astype(int)

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


def tune_parameters(
    plis,
    buyer_info,
    warm_ids,
    feature_cols,
    rolling_folds,
    scale_grid,
    cold_discount_grid,
    cold_topk_grid,
    savings_rate,
    total_fee,
    random_state=42,
):
    """Grid search parameters directly against average combined net score."""
    rng = np.random.RandomState(random_state)
    warm_list = np.array(sorted(list(warm_ids)))
    rng.shuffle(warm_list)

    pseudo_cold_size = max(8, int(len(warm_list) * 0.2))
    pseudo_cold_ids = set(warm_list[:pseudo_cold_size].tolist())
    warm_train_ids = warm_ids - pseudo_cold_ids
    warm_eval_ids = warm_ids - pseudo_cold_ids

    search_rows = []
    for scale, cdisc, ctopk in product(scale_grid, cold_discount_grid, cold_topk_grid):
        fold_metrics = []
        for fold in rolling_folds:
            fold_metrics.append(
                evaluate_fold(
                    plis=plis,
                    buyer_info=buyer_info,
                    warm_train_ids=warm_train_ids,
                    warm_eval_ids=warm_eval_ids,
                    pseudo_cold_ids=pseudo_cold_ids,
                    feature_cols=feature_cols,
                    fold=fold,
                    expected_spend_scale=scale,
                    cold_expected_discount=cdisc,
                    cold_top_k=ctopk,
                    savings_rate=savings_rate,
                    total_fee=total_fee,
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
    best = search_df.iloc[0]
    return {
        "best_scale": float(best["scale"]),
        "best_cold_discount": float(best["cold_discount"]),
        "best_cold_top_k": int(best["cold_top_k"]),
        "search_df": search_df,
        "warm_train_ids": warm_train_ids,
        "warm_eval_ids": warm_eval_ids,
        "pseudo_cold_ids": pseudo_cold_ids,
    }
