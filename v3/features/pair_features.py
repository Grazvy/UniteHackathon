import pandas as pd


def build_pair_features(df: pd.DataFrame, cutoff: pd.Timestamp, buyer_info: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (buyer, eclass) pair with compact RFM-like features."""
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
