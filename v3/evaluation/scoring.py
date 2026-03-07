from v3.config import SAVINGS_RATE, TOTAL_FEE


def realized_net(pred_df, truth_df):
    """Compute realized economic utility using one-time fee."""
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
