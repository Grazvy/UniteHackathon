# Feature Reference

Features computed per **(buyer × eclass)** pair from `plis_training.csv`.
All features are derived from the **feature window** (Jan 2023 – Dec 2024 for training,
Jan 2023 – Jun 2025 for final prediction).

---

## Volume features

| Feature | Formula | Intuition |
|---|---|---|
| `total_spend` | `sum(quantityvalue × vk_per_item)` | Total € spent by this buyer on this eclass. High spend = large savings potential if correctly predicted. |
| `n_orders` | Count of order lines | Raw order frequency. More orders = more recurring. |
| `monthly_avg` | `total_spend / span_months` | Average monthly spend — normalises for buyers observed over different time windows. |
| `last6m_spend` | `total_spend over last 6 months of window` | Recent spend signal. Captures whether demand is current or historical. |

---

## Frequency / recurrence features

| Feature | Formula | Intuition |
|---|---|---|
| `n_months_active` | Count of distinct calendar months with at least one order | Core recurrence signal. A buyer active in 12+ months is reliably recurring. |
| `activity_ratio` | `n_months_active / span_months` | Fraction of possible months with an order. 1.0 = ordered every month. |
| `span_months` | `(last_order − first_order).days / 30`, clipped ≥ 1 | How long the buyer has been purchasing this eclass. |

---

## Recency features

| Feature | Formula | Intuition |
|---|---|---|
| `recency_months` | `(cutoff − last_order).days / 30` | Months since last purchase. Low = recently active = likely to continue. |

---

## Stability features

| Feature | Formula | Intuition |
|---|---|---|
| `spend_cv` | `std(monthly_spend) / mean(monthly_spend)` per buyer×eclass | Coefficient of variation. Low CV = stable predictable demand = safer to predict as core. Falls back to 0 for single-month buyers. |

---

## Buyer-level features

| Feature | Formula | Intuition |
|---|---|---|
| `log_employees` | `log1p(avg_employees)` | Log-scaled buyer headcount from `estimated_number_employees`. Proxy for procurement volume and organisational size. |
| `buyer_total_spend` | `sum(line_value)` across all eclasses for this buyer | Buyer size signal. Large buyers tend to have more stable core demand. |

---

## Category-level features

| Feature | Formula | Intuition |
|---|---|---|
| `eclass_total_spend` | `sum(line_value)` across all buyers for this eclass | Category popularity signal. Widely purchased eclasses are more likely to be genuinely recurring across buyers. |

---

## Label definition

```
label = 1  if  (future_spend × SAVINGS_RATE) > (FEE × LABEL_MONTHS)
```

| Parameter | Default | Meaning |
|---|---|---|
| `SAVINGS_RATE` | `0.10` | Fraction of spend captured as savings (10 %) |
| `FEE` | `10.0` | Fixed monthly fee per predicted core demand eclass (€) |
| `LABEL_MONTHS` | `6` | Length of the label / evaluation window in months |

Break-even rule: a pair is labelled **core** only if future spend exceeds
`FEE × LABEL_MONTHS / SAVINGS_RATE = €600` in the label window.
