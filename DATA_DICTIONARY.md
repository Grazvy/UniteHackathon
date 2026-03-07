# Data Dictionary

---

## `customer_test.csv` — 100 rows × 5 cols
The **target buyers** you must submit Core Demand predictions for.

| Column | Type | Nulls | Description |
T|---|---|---|---|
| `legal_entity_id` | int | 0 % | Unique buyer identifier. Join key to `plis_training`. |
| `estimated_number_employees` | float | 2 % | Company headcount. Proxy for procurement volume. Heavily skewed (median 1,521 vs mean 19,064) — use log-transform as a feature. |
| `nace_code` | int | 0 % | Primary industry classification (EUROSTAT standard). Join key to `nace_codes`. **Stored as plain integer** here. |
| `secondary_nace_code` | float | 47 % | Optional secondary industry. Missing in ~half of buyers — treat as optional feature only. |
| `task` | str | 0 % | `"cold start"` (52 buyers, no usable history) or `"predict future"` (48 buyers, have PLIS history). Determines which pipeline to apply. |

---

## `nace_codes.csv` — 975 rows × 8 cols
Reference taxonomy to translate raw NACE codes into human-readable industry sectors.

| Column | Type | Nulls | Description |
|---|---|---|---|
| `nace_code` | int | 0 % | NACE code. Join key to `customer_test` and `plis_training`. |
| `n_nace_description` | str | 0 % | Full description of the specific NACE code. |
| `toplevel_section` | str | 0 % | Single-letter sector code (e.g. `A`, `C`, `G`). |
| `toplevel_section_description` | str | 0 % | Human-readable sector name (e.g. `Manufacturing`). **Best grouping for cold-start peer similarity.** |
| `nace_2digits` | float | 0 % | 2-digit NACE group (e.g. `28` = Manufacture of machinery). Use for finer cold-start sub-sector grouping, especially within Manufacturing (337 codes). |
| `nace_2digits_description` | str | 0 % | Description of the 2-digit group. |
| `nace_3digits` | float | 9 % | 3-digit NACE group. 88 top-level-only codes have no 3-digit entry — not a data error. |
| `nace_3digits_description` | str | 9 % | Description of the 3-digit group. |

> **Hierarchy:** `toplevel_section` → `nace_2digits` → `nace_3digits` → `nace_code`

---

## `plis_training.csv` — ~8.4M rows × 11 cols
**Core signal file.** One row = one order line item. Jan 2023 – Dec 2025.

| Column | Type | Nulls | Description |
|---|---|---|---|
| `orderdate` | date | 0 % | Date of the order (YYYY-MM-DD). Use to compute recency, frequency, and temporal splits. |
| `legal_entity_id` | int | 0 % | Buyer identifier. Join key to `customer_test`. |
| `set_id` | str | 8.1 % | Basket/order group ID — groups items ordered together. Not critical for Level 1; gaps are normal. |
| `sku` | str | 0 % | Specific product identifier. **Do not predict at this level** — too noisy and duplicate-prone. Aggregate to `eclass`. |
| `eclass` | float | 1.2 % | **Primary prediction target for Level 1.** 8-digit eCl@ss product category code (e.g. `27141104.0`). Multiple SKUs share the same eclass. Drop rows where null. Cast to `int → str` before use. |
| `manufacturer` | str | 0 % | Brand/supplier name. Core feature for Level 2. `"Unbekannt"` = unknown/unlabelled — exclude from Level 2. |
| `quantityvalue` | float | 0 % | Units ordered in this line. Median = 2, can reach 147,000 for bulk items. |
| `vk_per_item` | float | 0 % | Unit price in €. **Total line value = `quantityvalue × vk_per_item`.** Median ~€27, mean ~€60. |
| `estimated_number_employees` | float | 1.6 % | Buyer headcount at time of order. Consistent with `customer_test` column. |
| `nace_code` | float | 0.2 % | Buyer's primary NACE code. **Stored as float** (e.g. `3513.0`) — cast to `int → str` before joining to `nace_codes`. |
| `secondary_nace_code` | float | 49 % | Buyer's secondary NACE code. Sparse — use only as fallback. |

> **Key derived metric:** `total_spend = quantityvalue × vk_per_item`
> **Break-even rule:** Only predict an eclass if a buyer's total historical spend on it exceeds **€100** (10% savings rate × €100 = €10 = fee).

---

## `features_per_sku.csv` — ~18M rows × 5 cols
Product attribute catalogue in long format. One row = one feature of one SKU. Used for Level 3 clustering and SKU deduplication.

| Column | Type | Nulls | Description |
|---|---|---|---|
| `safe_synonym` | str | 0 % | **Normalised product name cluster.** SKUs with the same `safe_synonym` represent the same product concept. Use to deduplicate SKU variants before counting order frequency. Only 440 unique values in sample vs 59,351 SKUs — extreme deduplication potential. |
| `sku` | str | 0 % | Product SKU. Join key to `plis_training`. |
| `key` | str | 0 % | Feature attribute name (e.g. `Material`, `Schutzart` = IP protection class, `Nenndruck` = nominal pressure). 497 distinct keys in sample. |
| `fvalue` | str | 37 % | Specific feature value for this SKU (e.g. `nitrile`, `IP54`). Missing in 37 % of rows — always fall back to `fvalue_set`. |
| `fvalue_set` | str | 0.5 % | **Canonical/normalised feature value.** More stable than `fvalue` — use this as the primary value for Level 3 feature vectors. Practically always present. |

> **Usage pattern:** Pivot on `(sku, key) → fvalue_set` to build a feature matrix per SKU. Average 3.2 features per SKU, max 19.
