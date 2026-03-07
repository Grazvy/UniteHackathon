# data/

This folder contains data ingestion and buyer-level preprocessing.

## Files

- `loader.py`
  - Loads `customer_test.csv`, `nace_codes.csv`, and `plis_training.csv`.
  - Performs chunked transaction loading to avoid memory spikes.
  - Applies core type casting and line-value construction.
  - Creates `buyer_info` with NACE hierarchy and company size features.

## Why This Layer Exists

Separating loading from modeling prevents accidental data leakage and allows all downstream modules to work with consistent, typed DataFrames.

## Key Effects on Downstream Steps

- If eclass parsing is wrong here, feature builder and labels break everywhere.
- If buyer metadata encoding changes, both warm and cold logic are affected.
- If filtering test buyers is not done here, tuning metrics become misleading.
