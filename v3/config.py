from pathlib import Path
import pandas as pd

# Economic config
SAVINGS_RATE = 0.10
FEE = 10.0
LABEL_MONTHS = 6  # retained for context
TOTAL_FEE = FEE   # fee applied once per predicted element

# Time windows
FEATURE_CUTOFF = pd.Timestamp("2024-12-31")
LABEL_END = pd.Timestamp("2025-06-30")

ROLLING_FOLDS = [
    {
        "name": "2024H2",
        "train_end": pd.Timestamp("2024-06-30"),
        "val_start": pd.Timestamp("2024-07-01"),
        "val_end": pd.Timestamp("2024-12-31"),
    },
    {
        "name": "2025H1",
        "train_end": pd.Timestamp("2024-12-31"),
        "val_start": pd.Timestamp("2025-01-01"),
        "val_end": pd.Timestamp("2025-06-30"),
    },
]

# Search grids (net-score objective)
SCALE_GRID = [1.0, 1.2, 1.4]
COLD_DISCOUNT_GRID = [0.7, 0.8, 0.9, 0.95]
COLD_TOPK_GRID = [185, 200, 215, 230]

# Initial / fallback values
EXPECTED_SPEND_SCALE = 1.1
COLD_EXPECTED_DISCOUNT = 0.9
COLD_TOP_K = 200

# LightGBM params
LGBM_PARAMS_CLF = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}

LGBM_PARAMS_REG = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "unite"
OUT_DIR = ROOT
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
