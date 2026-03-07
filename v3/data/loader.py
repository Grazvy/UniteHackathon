import numpy as np
import pandas as pd


def load_core_data(data_dir):
    """Load customer, nace, and transactions data in one place.

    Returns:
        customers, nace, buyer_info, plis, warm_ids, cold_ids
    """
    customers = pd.read_csv(data_dir / "customer_test.csv", sep="\t")
    customers["nace_code"] = customers["nace_code"].astype(int)

    nace = pd.read_csv(data_dir / "nace_codes.csv", sep="\t")
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
        data_dir / "plis_training.csv",
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

    # keep only challenge buyers
    plis = plis[plis["legal_entity_id"].isin(set(customers["legal_entity_id"]))].copy()

    return customers, nace, buyer_info, plis, warm_ids, cold_ids
