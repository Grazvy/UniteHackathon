# viz/

This folder contains visualization utilities for model interpretability.

## Files

- `feature_importance.py`
  - `save_feature_importance_plots()`

## Produced Chart

`v3/plots/main_predictor_features.png` with three panels:
1. classifier split importance
2. regressor split importance
3. combined normalized importance

## Why This Matters

The pipeline uses two models with different roles.
A single importance chart can hide trade-offs, so this module renders model-specific and combined views to explain what drives final economic decisions.
