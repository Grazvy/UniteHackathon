# cold_start/

This folder handles cold-start candidate generation.

## Files

- `candidates.py`
  - `build_cold_candidates_from_warm()`

## What It Does

1. Takes warm predictions (with expected spend fields).
2. Aggregates sector-level statistics by `(nace_2digits, eclass)`.
3. Builds a global fallback by `eclass`.
4. For each cold buyer, creates candidate rows by copying sector/global means and filling buyer profile columns.

## Why This Is Separate

Cold-start logic is structurally different from warm scoring.
Keeping it isolated lets you improve candidate strategy (e.g., deeper NACE hierarchy, top-k logic, fallback policy) without touching warm modeling.

## Interaction With Other Layers

- Depends on feature schema from `features/schema.py`.
- Output rows are scored by `models/two_model.py`.
- Impacts `cold_proxy_net` in rolling tuning.
