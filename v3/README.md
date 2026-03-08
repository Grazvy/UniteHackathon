# v3 — Two-Stage Core Demand Prediction Pipeline

## Overview

All levels share a common **two-stage architecture** that separates warm (history-available) and cold (no-history) buyers:

1. **Stage 1 — Candidate Generation:** An `LGBMRanker` (lambdarank) scores all possible targets per buyer using only metadata features. For cold buyers, the top-K candidates are kept. Warm buyers skip this stage and use their historical pairs directly.

2. **Stage 2 — Value Estimation:** Two LightGBM models estimate economic value per candidate:
   - `clf` (classifier): predicts P(buy) — probability of future purchase
   - `reg` (regressor): predicts E(spend | buy) — expected spend given purchase (log-space)

   The core decision rule is: **include if `P(buy) × E(spend|buy) × SAVINGS_RATE > FEE`**

Validation uses **rolling temporal folds** (2024H2, 2025H1) with 20% of warm buyers held out as **pseudo-cold** to evaluate cold-start performance with known ground truth. A grid search over `EXPECTED_SPEND_SCALE` and `CANDIDATE_K_COLD` maximizes combined net € across folds.

---

## Files

| File | Description |
|------|-------------|
| `level1/level1.py` | Production Level 1 pipeline (eclass granularity) |
| `level1/two_stage_profit_model.py` | Earlier iteration of Level 1 (functionally equivalent, precursor to `level1.py`) |
| `level2/level2.py` | Production Level 2 pipeline with hierarchical L1 gating |
| `level2/2nd_level_adjusted.py` | Simpler Level 2 variant without L1 gating |
| `level3_feature_cluster_pipeline.py` | Level 3 pipeline using deterministic feature-based clustering |
| `optuna.py` | Optuna hyperparameter tuning for Level 1 (parameterized training with fold caching) |

---

## Architecture Differences Between Levels

### Level 1 (`level1/level1.py`)

**Prediction unit:** E-Class (functional product category, e.g. "office paper")

- Pair key: `(legal_entity_id, eclass)`
- Ranker trains on `(buyer, eclass)` pairs with popularity-weighted negative sampling (freq^0.75)
- Cold features: 7 numeric (buyer meta + eclass popularity) + `eclass_cat` (categorical)
- Cold value models add `rank_score` as an 8th numeric feature
- Warm features: 13 pair-history features (total_spend, n_orders, recency, monthly_avg, etc.)
- `TOTAL_FEE = FEE × 0.95` (discounted)
- No group-size caps on ranker training

### Level 2 (`level2/level2.py`)

**Prediction unit:** E-Class + Manufacturer (e.g. "office paper | HP")

Key changes from Level 1:

- **Composite key:** `item_key = f"{eclass}|{manufacturer}"` replaces `eclass` everywhere
- **Hierarchical L1 gating:** A lightweight Level-1 eclass ranker is trained first. For cold candidate generation, only items whose parent eclass falls in the buyer's top-K eclasses (default `L1_GATE_TOP_K=150`) are scored. This dramatically reduces the candidate space and false positives.
- **`l1_rank_score` as feature:** The L1 eclass ranker score is injected as an additional feature into both warm pair-history models and cold models (8 → 9 cold numeric features; 13 → 14 warm features)
- **Manufacturer deduplication:** After scoring, only `MAX_MANUFACTURERS_PER_ECLASS=20` items per (buyer, eclass) are kept, ranked by `expected_savings`
- **Ranker group-size cap:** `MAX_GROUP_SIZE_RANKER=9000` and `MAX_POS_PER_BUYER_RANKER=3000` prevent exceeding LightGBM's ~10k row-per-query limit (necessary because the eclass×manufacturer universe is much larger)
- **Higher `NEG_PER_POS=100`** (vs 20 in L1) to handle the sparser signal at finer granularity
- **`TOTAL_FEE = FEE`** (no discount)

### Level 2 Simplified (`level2/2nd_level_adjusted.py`)

Same as Level 2 but **without hierarchical L1 gating**:
- No L1 ranker trained, no `l1_rank_score` feature
- No eclass-level pre-filtering of candidates
- Lower `NEG_PER_POS=40`
- Otherwise identical composite-key architecture

### Level 3 (`level3_feature_cluster_pipeline.py`)

**Prediction unit:** E-Class + Feature Combination (deterministic cluster)

Key changes from Level 1/2:

- **Deterministic clustering:** Instead of manufacturer, products are grouped by a hash of their product feature profile. Uses `features_per_sku.csv` to extract key-value attributes per SKU.
- **Cluster ID construction:** `cluster_id = f"{eclass}|{sha1_hash(feature_signature)}"`. The signature is built from the top-K feature keys per eclass (by SKU support), taking the modal value per (SKU, key).
- **Feature key selection:** Top 6 keys per eclass, requiring ≥5% SKU support. Values are case-folded and normalized. SKUs without features fall back to `safe_synonym` or eclass-only clusters.
- **Pair key:** `(legal_entity_id, cluster_id)` replaces `(legal_entity_id, eclass)`
- **Same two-stage warm/cold architecture** as Level 1, but operating on `cluster_id` instead of `eclass`
- **Ranker group-size cap** same as Level 2 (`MAX_GROUP_SIZE_RANKER=9000`)
- **No hierarchical gating** (flat ranking over cluster universe)

### Optuna Tuning (`optuna.py`)

Operates at Level 1 granularity with:
- **Fold caching:** Precomputes all fold datasets once (pair features, ranker training data, cold training data). Trials only train models, avoiding redundant data prep.
- **Parameterized training functions:** All model params passed explicitly (no global state during tuning)
- **Tuning space:** LightGBM hyperparameters for all 5 models (warm_clf, warm_reg, ranker, cold_clf, cold_reg) + `expected_spend_scale` + `candidate_k_cold`
- **Different economic assumptions:** `SAVINGS_RATE=0.15`, `FEE=15.0` (exploring alternative scoring parameters)

---

## Shared Components

All levels share the same building blocks with level-specific adaptations:

| Component | Purpose |
|-----------|---------|
| `build_pair_features()` | Warm pair-history aggregates (spend, frequency, recency, etc.) |
| `train_ranker()` | LGBMRanker for cold candidate generation |
| `sample_negative_pairs()` | Popularity-weighted negative sampling for training |
| `train_cold_value_models()` | Cold P(buy) classifier + spend regressor using metadata + rank_score |
| `score_candidates_for_buyers()` | Score all targets per buyer, keep top-K |
| `infer_expected_spend_*()` | Apply decision rule: expected_savings > fee → is_core |
| `evaluate_fold()` | Rolling validation with warm eval + pseudo-cold eval |
| `realized_net()` | Compute net € score: Σ(savings) − Σ(fees) |

---

## Data Flow

```
plis_training.csv ──► feature engineering ──► (buyer, target) pairs
                                                    │
                              ┌──────────────────────┴──────────────────────┐
                              │                                             │
                         WARM BUYERS                                   COLD BUYERS
                              │                                             │
                    pair-history features                          Stage 1: LGBMRanker
                    (spend, freq, recency)                        scores all targets
                              │                                   keeps top-K candidates
                              │                                             │
                    warm_clf → P(buy)                            cold_clf → P(buy)
                    warm_reg → E(spend|buy)                      cold_reg → E(spend|buy)
                              │                                             │
                              └──────────────┬──────────────────────────────┘
                                             │
                                   P(buy) × E(spend|buy) × SAVINGS_RATE > FEE?
                                             │
                                      submission.csv
```

