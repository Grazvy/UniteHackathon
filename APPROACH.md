# Unite Hackathon – Data Insights & Approach Summary

## 1. Dataset Overview

| File | Rows | Key Columns |
|---|---|---|
| `customer_test.csv` | 100 | `legal_entity_id`, `nace_code`, `estimated_number_employees`, `task` |
| `nace_codes.csv` | 975 | `nace_code`, `toplevel_section_description`, `nace_2digits_description` |
| `plis_training.csv` | ~8.4 M | `orderdate`, `legal_entity_id`, `sku`, `eclass`, `manufacturer`, `quantityvalue`, `vk_per_item` |
| `features_per_sku.csv` | ~18 M | `sku`, `key`, `fvalue`, `fvalue_set`, `safe_synonym` |

> **Key column note:** `plis_training` has **no separate `value` column** – total order value must be computed as `quantityvalue × vk_per_item`.

---

## 2. Critical Dataset Observations

### Customers (`customer_test.csv`)
- **100 test buyers** to predict Core Demand for.
- **52 cold-start** (no or minimal history), **48 warm-start** (history available via PLIS).
- 47 % of buyers have NO secondary NACE code → industry enrichment needs careful handling.
- Dominant sector: **Manufacturing (44 %)**, followed by Electricity/Gas and Transport.
- Employee count spans 44 → 320,000 (heavily skewed; use log-scale or quantile bins).

### PLIS Training (`plis_training.csv`)
- Rich transaction history with **eclass** (E-class product category) and **manufacturer** per order line.
- `eclass` has ~1.2 % missing values → fill or exclude per strategy.
- `set_id` has 8 % missing → some orders lack a basket grouping.
- `manufacturer` is always present → strong signal for Level 2.
- `nace_code` in PLIS is stored as **float** (e.g. `3513.0`) – must be cast to integer string for joins.
- Revenue dominated by **Manufacturing** sector, followed by IT/Comms and Wholesale.
- Date range: use `orderdate` to distinguish warm-start buyers and compute recency/frequency metrics.

### Features per SKU (`features_per_sku.csv`)
- Long-format: one row per (SKU, feature key).
- Average ~3.2 distinct features per SKU (max 19).
- `fvalue` is missing in 37 % of rows → many features only carry a set-level value (`fvalue_set`).
- `safe_synonym` is a cluster-safe product name normalisation → essential for Level 3 deduplication.
- Use `key + fvalue_set` pairs as stable product descriptors, not raw `fvalue`.

---

## 3. Scoring Mechanics

```
Score = Σ savings(predicted) − Σ fees(predicted)
```

- **Savings** scale non-linearly with price & frequency (~√(price) × frequency).
- **Fee** is fixed per predicted Core Demand element, regardless of whether it was purchased.
- **Over-prediction** (recommend everything) inflates fees → net negative score.
- **Under-prediction** (recommend nothing) leaves savings on the table.
- **Target** the sweet spot: high-frequency, high-value, stable recurring needs.

---

## 4. Level 1 — E-Class Prediction (Recommended Starting Point)

**Goal:** For each buyer, predict which `eclass` codes represent recurring needs.

### Warm-Start Buyers (48 buyers with PLIS history)
1. Filter PLIS rows for each `legal_entity_id`.
2. Compute per-eclass metrics:
   - **Order frequency** (count of distinct order dates)
   - **Total spend** (`quantityvalue × vk_per_item` summed)
   - **Span** (months between first and last purchase)
   - **Recency** (months since last purchase)
3. Score each eclass with a composite signal, e.g.:
   ```
   score = log(1 + total_spend) × log(1 + order_frequency)
   ```
4. Apply a **threshold** that balances savings vs fees (tune on a holdout window using the last N months as pseudo ground truth).
5. **Recommended cutoffs:** eclass ordered in ≥ 2 distinct months AND total spend > some minimum (e.g. € 50).

### Cold-Start Buyers (52 buyers without history)
1. Map buyer's `nace_code` → sector via `nace_codes.csv`.
2. Compute **sector-level eclass frequency** from PLIS buyers in the same sector.
3. Recommend the top-K eclasses by sector (e.g. top 5–10 by order frequency in that sector).
4. Weight by employee size bucket (small/medium/large) if signal is strong.
5. Fallback: global top-K eclasses as a universal prior.

---

## 5. Level 2 — E-Class + Manufacturer Prediction

**Goal:** Refine predictions by adding brand specificity.

### Warm-Start Buyers
- Extend Level 1: for each retained eclass, identify the **dominant manufacturer(s)** used by that buyer.
- Keep only manufacturer combinations with ≥ 2 orders (avoid noise from one-off purchases).
- A buyer may have 1–2 manufacturers per eclass.
- Deduplicate by `(eclass, manufacturer)` pair.

### Cold-Start Buyers
- For each sector-recommended eclass, find the **most common manufacturer** across all sector peers.
- Use the top-1 or top-2 manufacturer per eclass.
- Be conservative — adding manufacturer specificity on cold-start increases fee risk.

### Pitfall: Duplicate SKUs
- Multiple SKUs from the same manufacturer map to the same eclass → count order lines correctly (group by `(eclass, manufacturer)`, not raw SKU).
- `features_per_sku.safe_synonym` can help consolidate SKU variants.

---

## 6. Level 3 — E-Class + Feature Combination (Clustered)

**Goal:** Construct stable product need clusters combining eclass + key product attributes.

### Feature Engineering Strategy
1. For each SKU in the buyer's history, retrieve its feature rows from `features_per_sku`.
2. Select the most **discriminative + stable** features:
   - Prefer `key` values with low cardinality and high coverage (e.g. material, size, packaging type).
   - Use `fvalue_set` as the canonical value when `fvalue` is missing.
3. Build a binary/categorical feature vector per SKU: `{eclass, key1=val1, key2=val2, ...}`.
4. **Cluster** SKUs with the same (eclass, feature subset) using string grouping or k-modes/DBSCAN on feature vectors.
5. Assign a `cluster_id` to each cluster.

### Clustering Approach
- Start with rule-based clustering: group by `(eclass, top_3_features_by_coverage)`.
- Optionally apply **MinHash LSH** or **TF-IDF + cosine similarity** on feature text for fuzzier deduplication.
- Ensure each cluster maps to a coherent, human-interpretable product type.

### Portfolio Discipline
- Merge clusters that represent the same underlying need but differ only in irrelevant attributes.
- Avoid creating one cluster per feature combination variant — this multiplies fees without adding savings.

---

## 7. Portfolio Construction & Optimisation

Regardless of level, apply portfolio filtering before submitting:

1. **Recency filter:** Drop any eclass/manufacturer/cluster not purchased in the last X months (warm-start).
2. **Frequency floor:** Require ≥ N distinct order events per year to justify a fixed monthly fee.
3. **Value floor:** Rough estimate: monthly fee F is covered when E[monthly_savings] > F. Since savings ≈ C × √(avg_price) × monthly_frequency, set a minimum spend threshold.
4. **Deduplication:** For warm-start, collapse SKU variants representing the same functional need into a single prediction row.

Rule of thumb: **quality over quantity** — 5 strong predictions beat 20 weak ones.

---

## 8. Submission Format

```csv
buyer_id,predicted_id
```

- **Level 1:** `predicted_id` = `eclass_id` (e.g. `27141104`)
- **Level 2:** `predicted_id` = `eclass_manufacturer_id` (e.g. `27141104_WAGO Kontakttechnik`)
- **Level 3:** `predicted_id` = `cluster_id` (e.g. `cluster_042`)

Multiple rows per buyer are allowed and expected.

---

## 9. Suggested Development Sequence

```
1. Build warm-start Level 1 pipeline (frequency + spend threshold)
2. Build cold-start Level 1 via sector similarity
3. Validate on pseudo-holdout (last 3 months of PLIS)
4. Extend to Level 2 by adding manufacturer layer
5. Explore Level 3 feature extraction and clustering for bonus points
6. Tune portfolio threshold to maximise Score = savings − fees
```

---

## 10. Quick Reference: Important Column Names

| Dataset | Column | Notes |
|---|---|---|
| plis | `eclass` | E-class category code (Level 1) |
| plis | `manufacturer` | Brand (Level 2) |
| plis | `quantityvalue` | Units ordered |
| plis | `vk_per_item` | Unit price (€) |
| plis | `nace_code` | Float – cast to `int → str` before joins |
| features | `safe_synonym` | Normalised product name cluster |
| features | `key` | Feature attribute name |
| features | `fvalue_set` | Preferred stable feature value |
| customers | `task` | `"cold start"` or `"predict future"` |
| nace | `toplevel_section_description` | Sector label for cold-start grouping |
