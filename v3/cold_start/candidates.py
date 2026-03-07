import pandas as pd


def build_cold_candidates_from_warm(warm_pred, buyer_info, cold_ids, feature_cols, cold_top_k):
    """Create cold candidate rows by sector-level feature imputation."""
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
