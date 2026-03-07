# v3 Modular Pipeline

This folder contains a modular rewrite of the `v2_unified/two_model_simple.py` approach.

## Why v3 Exists

The original script is useful for experimentation, but difficult to maintain because data loading, feature engineering, model training, validation, cold-start logic, plotting, and submission writing are all in one file.

`v3/` separates those concerns into focused modules so you can:
- test pieces independently,
- tune safely without side effects,
- explain and debug each step faster,
- reuse components for Level 2/3 later.

## Pipeline Flow

1. `data.loader` reads raw CSVs and prepares buyer metadata.
2. `features.pair_features` builds `(buyer, eclass)` feature rows.
3. `evaluation.rolling` tunes scale/discount/top-k on rolling windows using net score.
4. `models.two_model` trains classifier + regressor and runs expected-spend inference.
5. `cold_start.candidates` creates cold-start candidates from warm sector behavior.
6. `evaluation.scoring` computes economic net utility metrics.
7. `viz.feature_importance` generates feature importance chart.
8. `io.submission` assembles `buyer_id,predicted_id` output.
9. `run_pipeline.py` orchestrates everything.

## How to Run

```bash
/Users/sarpsahinalp/projects/UniteHackathon/.venv/bin/python v3/run_pipeline.py
```

Outputs:
- `submission.csv` at repo root
- feature importance plot in `v3/plots/main_predictor_features.png`

## Design Notes

- Fee logic is one-time fee per predicted element (`TOTAL_FEE = FEE`).
- Rolling folds match your requested windows (`2024H2` and `2025H1`).
- Tuning objective is direct average combined net utility.
- Warm and cold-proxy nets are tracked separately during tuning.
