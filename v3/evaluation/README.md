# evaluation/

This folder contains net-score-based evaluation and rolling validation logic.

## Files

- `scoring.py`
  - `realized_net()`: computes earned savings, fees, net score, TP, FP.
- `rolling.py`
  - `evaluate_fold()`: trains on one history window and evaluates on one forward window.
  - `tune_parameters()`: grid-searches scale/discount/top-k using average combined net.

## Why This Layer Is Critical

Accuracy metrics alone are not sufficient for this challenge.
The target is economic utility:

`Net = Sum(predicted * future_spend * SAVINGS_RATE) - Sum(predicted * TOTAL_FEE)`

By centralizing this logic here, model tuning aligns directly with the leaderboard objective.

## Warm vs Cold-Proxy Tracking

Rolling evaluation stores:
- `warm_net`
- `cold_proxy_net`
- `combined_net`

This prevents warm gains from masking cold regressions (and vice versa).
