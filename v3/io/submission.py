import pandas as pd


def build_submission_df(warm_pred, cold_pred=None):
    """Build challenge-format submission with buyer_id,predicted_id."""
    warm_sub = (
        warm_pred.loc[warm_pred["is_core"] == 1, ["legal_entity_id", "eclass"]]
        .rename(columns={"legal_entity_id": "buyer_id", "eclass": "predicted_id"})
    )

    if cold_pred is None or cold_pred.empty:
        cold_sub = pd.DataFrame(columns=["buyer_id", "predicted_id"])
    else:
        cold_sub = (
            cold_pred.loc[cold_pred["is_core"] == 1, ["legal_entity_id", "eclass"]]
            .rename(columns={"legal_entity_id": "buyer_id", "eclass": "predicted_id"})
        )

    submission = (
        pd.concat([warm_sub, cold_sub], ignore_index=True)
        .sort_values(["buyer_id", "predicted_id"])
        .reset_index(drop=True)
    )
    return submission, warm_sub, cold_sub
