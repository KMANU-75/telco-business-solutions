# Customer Analytics for a Telecom Company

**Predicting Revenue, Churn, and Customer Segments for FreshWave Telecom**

Data Science Capstone (2026) · Regression · Classification · K-Means Clustering

---

## Project Overview

FreshWave Telecom wants to turn its historical customer data into decisions for three teams. This project answers all three questions from a single dataset of ~7,000 customers:

- **Finance** — estimate the revenue a customer generates from their profile and subscribed services.
- **Retention** — flag customers likely to cancel (churn) so they can be contacted proactively.
- **Marketing** — group customers into meaningful segments for tailored campaigns instead of generic promotions.

**Dataset:** the public "Telco Customer Churn" dataset (Kaggle, by blastchar) — 7,043 customers and 21 columns covering demographics, account details, subscribed services, and the churn label.

---

## Repository Structure

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── FINAL_PROJECT WORK.ipynb   # data prep + all three tasks
├── requirements.txt
└── README.md
```

---

## Data Preparation

Decisions made before modelling, with reasoning:

- **TotalCharges type fix:** the column loaded as text because a handful of rows held blank strings. It was converted to numeric, and the resulting missing values were filled with 0.0 — these correspond to brand-new customers who have not yet accrued a total charge.

- **Dropped identifier:** customerID was removed. It is a unique key with zero predictive value and only adds noise/dimensionality.

- **Categorical encoding:** categorical variables were one-hot encoded with pandas `get_dummies` (drop_first=True) to avoid implying any false ordering and to avoid the dummy-variable trap.

- **Scaling:** features were standardised with `StandardScaler` for the models that are sensitive to feature scale (Logistic Regression and K-Means).

- **Train/test split:** an 80/20 split with random_state=42 was used throughout for a fair, reproducible evaluation.

---

## Task 1 — Predicting Revenue (Regression)

### Approach

The Finance team's "revenue" question was framed as predicting **MonthlyCharges** — the recurring monthly bill a customer generates — from their demographic profile and the services they subscribe to. To prevent target leakage, both TotalCharges and Churn were excluded from the feature set (TotalCharges is essentially monthly charge × tenure, so it would give the answer away). A **Linear Regression** model was trained on the one-hot-encoded features.

### Results

| Metric | Value | Business Meaning |
|--------|-------|------------------|
| MAE | $0.79 | On average, predictions differ from the actual monthly bill by only 79 cents. |
| RMSE | $1.05 | Errors are tight and consistent; extreme mispredictions are virtually non-existent. |
| R² | 0.9988 | The features explain 99.88% of the variation in monthly billing. |

### What This Means for Finance

Across monthly charges ranging from $18.25 to $118.75, the model predicts a customer's bill to within about 79 cents on average. This is effectively a precise pricing map: each service adds a predictable amount to the bill. The model is robust enough to support monthly revenue forecasting and to auto-generate dynamic quote estimates for new subscription packages.

---

## Task 2 — Predicting Customer Churn (Classification)

### Approach

The target is the Churn column (mapped to 1 = Yes, 0 = No). Because churn is imbalanced (~27% churn), accuracy alone is misleading, so **recall on the churn class was treated as the primary business metric** — the Retention team cares most about catching customers who are actually about to leave. A **Logistic Regression** model was used with `class_weight='balanced'` to counter the imbalance, on standardised, one-hot-encoded customer-profile features.

### Results

| Metric | Value | Note |
|--------|-------|------|
| Recall | 82.31% | Primary metric — flags 8 out of every 10 real churners. |
| Precision | 52.48% | About half of flagged customers would not have churned. |
| Accuracy | ~75.6% | Secondary — not the focus given class imbalance. |
| ROC-AUC | ~0.86 | Strong separation between churners and non-churners. |

### What This Means for Retention

Logistic Regression with balanced class weights was selected. Its **82.31% recall** means it successfully flags 8 out of 10 potential churners. The precision of **52.48%** means roughly half of the flagged customers would not have churned — an acceptable trade-off, because offering a discount or incentive to a loyal customer costs far less than losing a customer entirely. Handed to the team tomorrow, this model would give them a reliable, high-sensitivity proactive call list.

---

## Task 3 — Segmenting Customers (K-Means)

### Approach

Customers were grouped with K-Means clustering on standardised, one-hot-encoded features, deliberately excluding the Churn label so the segments reflect natural customer groups rather than a churn model. Four segments (k=4) were used, and each cluster was profiled on tenure, monthly and total charges, its most common contract and internet type, and its observed churn rate to turn the maths into marketing personas.

### The Four Segments

| Segment | Size | Avg Tenure | Avg Monthly | Avg Lifetime | Top Contract / Internet | Churn |
|---------|------|------------|-------------|--------------|------------------------|-------|
| 0 — Mid-tenure DSL | 682 | 32 mo | $42 | $1,496 | Month-to-month / DSL | 25% |
| 1 — New High-Spend At-Risk | 2,735 | 16 mo | $74 | $1,172 | Month-to-month / Fiber | 47% |
| 2 — Loyal High-Value | 2,100 | 56 mo | $92 | $5,152 | Two year / Fiber | 15% |
| 3 — Loyal Low-Cost | 1,526 | 31 mo | $21 | $663 | Two year / None | 7% |

### Segment Profiles and Campaign Ideas

**Segment 0 — Mid-tenure DSL**  
Moderate-tenure, mid-spend DSL customers on month-to-month plans with a middling 25% churn rate. *Campaign:* encourage contract commitment and upsell fibre/add-ons to lift value and lock them in.

**Segment 1 — New High-Spend At-Risk**  
The largest group: newer fibre customers on month-to-month plans with an alarming ~47% churn rate. This is the biggest opportunity — high value and high flight risk. *Campaign:* a contract-lock incentive protects the most revenue and is the top priority.

**Segment 2 — Loyal High-Value**  
Long-tenured, high-spend, two-year fibre customers and the most valuable group (~$5,150 lifetime, only 15% churn). *Campaign:* VIP loyalty rewards and retention perks to protect this revenue base.

**Segment 3 — Loyal Low-Cost**  
Long-relationship, very low-spend customers (often no internet service) and the stickiest group (7% churn). *Campaign:* low-touch retention with targeted upsell of internet/streaming to grow their small footprint.

---

## Key Findings & Business Recommendations

**Finance:**  
Monthly revenue is almost perfectly predictable from a customer's service mix (MAE $0.79, R² 0.9988). Ready to deploy for revenue forecasting and instant quote generation.

**Retention:**  
Churn is predictable ahead of time (ROC-AUC 0.86). The balanced Logistic Regression model catches ~82% of would-be churners — a ready-to-use proactive call list, accepting the ~48% false-alarm rate as a worthwhile trade-off.

**Marketing:**  
Customers split into four clear segments. Prioritise Segment 1 (new high-spend, ~47% churn) with a contract-lock incentive, protect Segment 2 (loyal high-value), and grow the low-cost loyal groups through targeted upsell.

---

## Reproducing This Project

### Prerequisites

```bash
pip install -r requirements.txt
```

**Core stack:** Python 3, pandas, numpy, scikit-learn. Open `FINAL_PROJECT WORK.ipynb` and run the cells top to bottom. All models use `random_state=42` and an 80/20 train/test split, so results are reproducible.

---

## Author

HENRY NEWWELL KOFI ANKU — Digital World Tech Academy (2026)

---

*This portfolio project demonstrates supervised regression (revenue prediction), supervised classification (churn detection), and unsupervised clustering (customer segmentation) on real-world telecom data. All decisions — feature engineering, model selection, evaluation metrics — are justified and defensible in an interview context.*
