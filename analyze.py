"""
Analysis script for the Unite Hackathon datasets.

Files:
  - customer_test.csv   : ~100 rows  – customers with their NACE codes and task type
  - nace_codes.csv      : ~975 rows  – NACE code reference table
  - plis_training.csv   : ~8.4 M rows – order/transaction history (sampled for speed)
  - features_per_sku.csv: ~18 M rows  – product feature attributes (sampled for speed)
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "unite")

# ── helpers ────────────────────────────────────────────────────────────────────

def load(filename: str, sep: str = "\t", nrows: int | None = None) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def basic_info(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    print(f"\n  Null counts:")
    nulls = df.isnull().sum()
    for col, n in nulls.items():
        pct = n / len(df) * 100
        print(f"    {col:<35} {n:>8,}  ({pct:.1f} %)")
    print(f"\n  Preview (first 3 rows):")
    print(df.head(3).to_string(index=False))


# ── load data ──────────────────────────────────────────────────────────────────

section("Loading datasets")

customers = load("customer_test.csv")
nace      = load("nace_codes.csv")

# Large files – load a sample to keep analysis fast
SAMPLE = 200_000
print(f"  (Loading {SAMPLE:,}-row sample from the two large files)")
plis      = load("plis_training.csv",   nrows=SAMPLE)
features  = load("features_per_sku.csv", nrows=SAMPLE)


# ── basic info ─────────────────────────────────────────────────────────────────

section("Basic information")
basic_info(customers, "customer_test.csv")
basic_info(nace,      "nace_codes.csv")
basic_info(plis,      "plis_training.csv (sample)")
basic_info(features,  "features_per_sku.csv (sample)")


# ── customer_test analysis ─────────────────────────────────────────────────────

section("Customer Test – exploratory analysis")

print("\n  Task distribution:")
print(customers["task"].value_counts().to_string())

print("\n  Estimated employees – summary stats:")
emp = pd.to_numeric(customers["estimated_number_employees"], errors="coerce")
print(emp.describe().to_string())

print(f"\n  Unique primary NACE codes  : {customers['nace_code'].nunique()}")
print(f"  Customers with secondary NACE: "
      f"{customers['secondary_nace_code'].notna().sum()} / {len(customers)}")

# Enrich with NACE descriptions
nace_lookup = nace[["nace_code", "toplevel_section_description", "nace_2digits_description"]].copy()
nace_lookup["nace_code"] = nace_lookup["nace_code"].astype(str).str.strip()
customers["nace_code"] = customers["nace_code"].astype(str).str.strip()

customers_enriched = customers.merge(nace_lookup, on="nace_code", how="left")

print("\n  Top 10 NACE sectors (customer_test):")
sector_counts = (
    customers_enriched["toplevel_section_description"]
    .value_counts()
    .head(10)
)
print(sector_counts.to_string())


# ── NACE codes – reference analysis ───────────────────────────────────────────

section("NACE Codes – reference table")

print(f"\n  Unique top-level sections : {nace['toplevel_section'].nunique()}")
print(f"  Unique 2-digit groups     : {nace['nace_2digits'].nunique()}")
print(f"  Unique 3-digit groups     : {nace['nace_3digits'].nunique()}")
print(f"  Total NACE codes          : {len(nace)}")

print("\n  Codes per top-level section:")
print(nace.groupby("toplevel_section_description")["nace_code"].count().to_string())


# ── plis_training analysis ─────────────────────────────────────────────────────

section(f"PLIS Training (sample: {SAMPLE:,} rows) – exploratory analysis")

plis["orderdate"]   = pd.to_datetime(plis["orderdate"],   errors="coerce")
plis["quantityvalue"] = pd.to_numeric(plis["quantityvalue"], errors="coerce")
plis["vk_per_item"]   = pd.to_numeric(plis["vk_per_item"],   errors="coerce")
plis["total_value"]   = plis["quantityvalue"] * plis["vk_per_item"]

print(f"\n  Date range          : {plis['orderdate'].min()} → {plis['orderdate'].max()}")
print(f"  Unique customers    : {plis['legal_entity_id'].nunique():,}")
print(f"  Unique SKUs         : {plis['sku'].nunique():,}")
print(f"  Unique sets         : {plis['set_id'].nunique():,}")
print(f"  Unique manufacturers: {plis['manufacturer'].nunique():,}")

print("\n  Total order value (qty × unit price) – summary stats (€):")
print(plis["total_value"].describe().to_string())

print("\n  Quantity – summary stats:")
print(plis["quantityvalue"].describe().to_string())

print("\n  Top 10 manufacturers by order count:")
print(plis["manufacturer"].value_counts().head(10).to_string())

print("\n  Top 10 NACE codes in training orders:")
print(plis["nace_code"].value_counts().head(10).to_string())

monthly = (
    plis.dropna(subset=["orderdate"])
    .set_index("orderdate")
    .resample("ME")["total_value"]   # month-end
    .agg(["count", "sum", "mean"])
    .rename(columns={"count": "orders", "sum": "total_value", "mean": "avg_value"})
)
print("\n  Monthly order summary (sample):")
print(monthly.to_string())


# ── features_per_sku analysis ──────────────────────────────────────────────────

section(f"Features per SKU (sample: {SAMPLE:,} rows) – exploratory analysis")

print(f"\n  Unique SKUs             : {features['sku'].nunique():,}")
print(f"  Unique feature keys     : {features['key'].nunique():,}")
print(f"  Unique safe_synonyms    : {features['safe_synonym'].nunique():,}")

print("\n  Top 15 most common feature keys:")
print(features["key"].value_counts().head(15).to_string())

features_per_sku = features.groupby("sku")["key"].nunique()
print("\n  # of distinct features per SKU – summary:")
print(features_per_sku.describe().to_string())


# ── cross-file: join plis ↔ nace ──────────────────────────────────────────────

section("Cross-file: PLIS orders enriched with NACE sector")

# PLIS nace_code arrives as float (e.g. 3513.0) – normalise to plain integer string
plis["nace_code"] = (
    pd.to_numeric(plis["nace_code"], errors="coerce")
    .apply(lambda x: str(int(x)) if pd.notna(x) else None)
)
plis_enriched = plis.merge(nace_lookup, on="nace_code", how="left")

print("\n  Revenue by top-level sector (sample):")
rev_by_sector = (
    plis_enriched.groupby("toplevel_section_description")["total_value"]
    .agg(orders="count", total_revenue="sum", avg_order_value="mean")
    .sort_values("total_revenue", ascending=False)
)
print(rev_by_sector.to_string())

print("\n  Top 10 SKUs by total quantity ordered (sample):")
top_skus = (
    plis.groupby("sku")["quantityvalue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_skus.to_string())

print("\nAnalysis complete.")
