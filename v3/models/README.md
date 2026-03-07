# models/

This folder contains the two-model learning logic.

## Files

- `two_model.py`
  - `train_two_models()`: trains LightGBM classifier + regressor
  - `infer_expected_spend()`: combines outputs into expected spend and economic decision fields

## Model Design

- Classifier target: `buy_label` (whether future spend > 0)
- Regressor target: `log1p(future_spend)` on positive-spend rows
- Combination:
  - `expected_spend = p_buy * spend_if_buy * expected_spend_scale`
  - `expected_savings = expected_spend * SAVINGS_RATE`
  - core if `expected_savings > TOTAL_FEE`

## Why Separate Models

The purchase/no-purchase boundary and spend magnitude behave differently statistically. Splitting them improves stability versus a single direct target.
