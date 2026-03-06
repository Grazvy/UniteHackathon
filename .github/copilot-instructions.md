

# **Hackathon Challenge: Core Demand Prediction & Value Optimization**

## **Mission**

Your task is to predict Core Demand for a set of buyers. This challenge is an economic portfolio optimization problem: you must recommend a set of recurring procurement needs that maximizes Net Economic Benefit (€).

In many organizations, procurement follows a typical demand curve: a relatively small set of items or functional needs accounts for a large share of purchasing volume and operational effort, while a long tail of irregular purchases occurs less frequently.

The goal of this challenge is to identify those recurring and economically relevant needs and separate them from the long tail of ad-hoc demand.

Participants will encounter two situations:

Cold Start – Buyers with little or no transaction history. Predictions must rely on market signals, buyer similarities, industry patterns, or collective demand patterns.

Warm Start – Buyers with transaction history available. Historical purchasing patterns can be used to infer recurring needs.

The challenge is not simply predicting what buyers purchase, but identifying which needs are stable and economically meaningful enough to justify inclusion in Core Demand.

---

## **Context**

Unite will launch the Core Demand feature in Q1 2026.

The feature introduces a structural separation between two types of procurement.

### **Core Demand**

Core Demand represents recurring, high-relevance procurement needs.

Buyers commit these needs into a Core Demand portfolio. Each element included in the portfolio triggers a monthly fixed fee. In return the buyer benefits from improved sourcing conditions and lower processing costs.

Core Demand therefore focuses on the part of procurement that is predictable and recurring.

### **Long Tail Demand**

Long Tail Demand represents infrequent or ad-hoc purchases.

These purchases occur irregularly, are harder to predict, and do not justify a recurring fee. They continue to be handled through standard spot-buy or marketplace mechanisms.

Because Core Demand involves a recurring fee per included element, selecting the right portfolio becomes an economic optimization problem.

---

## **The Optimization Problem**

Each predicted Core Demand element creates a trade-off.

Benefit (Savings)

If a recurring need is correctly identified as Core Demand, the buyer benefits from improved sourcing terms and lower operational costs.

Cost (Fee)

Every element included in the Core Demand list triggers a fixed monthly fee.

If the item is not purchased frequently enough, the fee may outweigh the savings generated.

The objective is therefore to construct an optimal Core Demand portfolio per buyer.

Recommending everything leads to excessive fees, while recommending nothing misses potential savings.

---

## **Prediction Levels**

Predictions can be submitted at three different abstraction levels.

Higher levels allow more precise targeting but increase complexity and noise sensitivity.

### **Level 1 — E-Class**

Predict recurring needs at the E-Class level.

E-Class represents a functional product category independent of specific suppliers or SKUs.

Examples may include office paper, nitrile gloves, printer toner, or industrial cleaning agents.

Advantages include robustness against duplicate SKUs and stable categorization.

---

### **Level 2 — E-Class + Manufacturer**

Predict recurring needs at the E-Class + Manufacturer level.

This adds brand or supplier specificity to the functional category.

Example:

```
E-Class: Office Paper
Manufacturer: HP
```

This level provides higher precision and can capture brand preferences, but it is more sensitive to sparse data and catalog noise.

---

### **Level 3 — E-Class + Feature Combination (Clustered)**

Level 3 allows participants to define product needs using E-Class combined with a set of product features.

Examples of features may include size, material, packaging, norms, or performance attributes.

Example representation:

```
E-Class: Protective Gloves
Features: nitrile + powder-free + size L
```

Because product catalogs may contain duplicates or inconsistent attributes, predicted feature combinations should be clustered into stable groups.

Level 3 focuses primarily on the methodological approach rather than pure score performance.

---

## **Data**

Participants will work with raw anonymized procurement data.

Important characteristics of the dataset include:

- duplicate products
- missing attributes
- inconsistent product descriptions
- multiple SKUs representing the same functional need

This reflects typical real-world procurement data.

Participants are encouraged to perform feature engineering, product normalization, clustering, and category abstraction.

Dataset documentation will be provided separately.

---

## **Scoring (Euro Score)**

Submissions are evaluated using a scoring script that compares predictions with hidden ground truth (actual purchases after a cutoff date).

Conceptually the score follows:

Score = Sum(Savings) - Sum(Fees)

Savings are generated when predicted Core Demand elements correspond to recurring purchases.

Savings scale with economic relevance, which typically depends on purchase price, frequency, and volume. In practice savings often scale non-linearly, for example approximately with the square root of purchase price combined with demand frequency.

Fees represent fixed monthly costs per predicted Core Demand element.

The scoring function should be treated as a black box. Internal parameters may change, therefore solutions should be robust rather than tuned to a single assumed formula.

The final result is the total economic utility across all buyers.

---

## **Strategic Hints**

Use historical signals where available. Warm-start buyers often reveal recurring needs clearly through purchase frequency and volume.

For cold-start buyers, industry patterns and similarities between buyers can provide useful signals.

Avoid predicting exact SKUs whenever possible. Functional abstractions such as E-Class or clustered needs are typically more robust.

Portfolio discipline is important. Recommending multiple variations of the same need increases fees without necessarily increasing savings.

---

## **Deliverables**

### **submission.csv**

A CSV file containing predictions:

```
buyer_id,predicted_id
```

The meaning of predicted_id depends on the prediction level:

- Level 1 → eclass_id
- Level 2 → eclass_manufacturer_id
- Level 3 → cluster_id

Multiple rows per buyer are allowed.

### **Code Bundle**

A reproducible repository containing data preparation, feature engineering, modeling approach, and prediction pipeline.

### **Short Report**

A short report (1–2 pages) explaining the modeling approach, how cold and warm buyers were handled, clustering or abstraction strategies, and how the portfolio trade-off between fees and savings was addressed.

---

## **Rating Criteria**

Evaluation will consider three aspects.

### **Level 1 Score**

Total Euro Score achieved using E-Class predictions.

### **Level 2 Score**

Total Euro Score achieved using E-Class + Manufacturer predictions.

### **Level 3 Approach**

Level 3 will be evaluated based on the quality of the methodology, including feature engineering, clustering strategy, and economic reasoning behind portfolio construction.