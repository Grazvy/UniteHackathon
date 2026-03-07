# features/

This folder defines the feature schema and pair-level feature builder.

## Files

- `schema.py`
  - Central list of model input columns (`FEATURE_COLS`).
- `pair_features.py`
  - Builds one feature row per `(legal_entity_id, eclass)`.

## What the Builder Computes

- Spend magnitude: `total_spend`, `monthly_avg`
- Frequency/activity: `n_orders`, `n_months_active`
- Recency: `recency_months`, `last6m_spend`
- Scale context: `buyer_total_spend`, `eclass_total_spend`
- Buyer profile: `log_employees`, `section_enc`, `nace_2digits`, `has_secondary_nace`

## Why It Matters

These features feed both models:
- classifier (buy/no-buy probability),
- regressor (spend if buy).

Any feature drift here changes both warm and cold outcomes, because cold-start imputation reuses this schema.
