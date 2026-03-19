# ⚖️ DriftSafe — Algorithmic Bias & Drift Monitoring Dashboard

A post-deployment fairness monitoring system for loan approval ML models. DriftSafe
detects when a frozen production model becomes unfair over time due to data drift —
with **no model changes** — using Disparate Impact and Approval Gap across multiple
sensitive attributes and two geographic regions (India & USA).

---

## 🏗️ Architecture

```
India CSV / USA Excel
       │
       ▼
Feature Engineering  ──►  Logistic Regression (trained on T0, FROZEN)
       │                          │
       ▼                          ▼
Time Windows (T0→T3)  ──►  Fairness Monitor
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             Disparate      Approval      Alert Level
              Impact          Gap        (OK/WARN/ALERT)
                    └─────────────┴─────────────┘
                                  │
                                  ▼
                         Flask API ──► HTML Dashboard
                         (india.html / usa.html)
```

**Key design principle:** The model is trained **once** on the baseline period (T0)
and never retrained. All fairness degradation seen in T1–T3 is caused purely by data
drift — demonstrating why continuous post-deployment monitoring is essential.

---

## ⚙️ Tech Stack

| Component        | Technology                                  |
|------------------|---------------------------------------------|
| Language         | Python 3.11                                 |
| ML Model         | scikit-learn — LogisticRegression           |
| Data             | pandas, NumPy, openpyxl                     |
| Backend API      | Flask                                       |
| Dashboard        | HTML / CSS / Vanilla JS (no framework)      |
| Fairness Metrics | Custom implementation (DI, Approval Gap)    |

---

## 📦 Datasets

| Region | File                           | Records | Features | Date Range    | Label          |
|--------|--------------------------------|---------|----------|---------------|----------------|
| India  | `IndianData_final.csv`         | 4,269   | 21       | 2018 – 2024   | Proxy rule     |
| USA    | `Dated_Loan_Approval_Data.xlsx`| 50,000  | 21       | 2015 – 2025   | `loan_status`  |

**India proxy label:** credit_score ≥ 650 AND debt_to_income_ratio ≤ 0.40  
**USA actual label:** `loan_status` column (1 = approved, 0 = rejected)  
**USA baseline approval rate:** 55.0% | **USA model accuracy on T0:** 82.4%

---

## 📊 Results

### Fairness Metric Definitions

| Metric | Formula | Threshold | Meaning |
|--------|---------|-----------|---------|
| **Disparate Impact (DI)** | P(approval\|protected) / P(approval\|reference) | ≥ 0.80 (80% rule) | Core fairness signal |
| **Approval Gap** | \|rate_protected − rate_reference\| | < 0.10 | Absolute disparity |

---

### 🇮🇳 India — 4,269 Records (~1,067 per window)

#### Quick Test (standalone — `python bias_quick_test.py`)

| Metric | Before Drift | After Drift |
|--------|-------------|-------------|
| Approval Rate — Young (≤35) | 24.9% | 16.6% |
| Approval Rate — Senior (>35) | 25.7% | 25.7% |
| **Disparate Impact** | **0.9696 ✅** | **0.6464 🚨** |
| Approval Gap | 0.008 | 0.091 |
| 80% Rule | PASS | **VIOLATION** |

DI dropped **−0.32** after threshold tightening. The 80% threshold was crossed — fairness alert triggered.

---

#### Time-Series — India Age Group (Young vs Middle)

| Period | Young Approval | Middle Approval | DI    | Gap   | Status |
|--------|---------------|-----------------|-------|-------|--------|
| T0     | 22.3%         | 25.7%           | 0.867 | 0.034 | ⚠️ Baseline |
| T1     | 12.1%         | 20.8%           | 0.582 | 0.087 | 🚨 ALERT |
| T2     | 16.1%         | 25.0%           | 0.646 | 0.088 | 🚨 ALERT |
| T3     | 14.0%         | 24.4%           | 0.575 | 0.104 | 🚨 ALERT |

#### Time-Series — India Income Segment (Low vs Medium)

| Period | Low (<40K) | Medium (40-80K) | DI    | Gap   | Status |
|--------|-----------|-----------------|-------|-------|--------|
| T0     | 25.6%     | 22.6%           | 1.136 | 0.031 | ✅ FAIR |
| T1     | 13.5%     | 20.6%           | 0.654 | 0.071 | 🚨 ALERT |
| T2     | 13.8%     | 23.7%           | 0.584 | 0.099 | 🚨 ALERT |
| T3     | 12.9%     | 22.5%           | 0.573 | 0.096 | 🚨 ALERT |

> India key finding: Income segment starts **fair at T0 (DI=1.136)** then collapses
> to **0.573 by T3** — a drop of 0.56 — with zero model changes.

---

### 🇺🇸 USA — 50,000 Records (~12,500 per window)

#### Time-Series — USA Age Group (Young vs Middle)

| Period | Young Approval | Middle Approval | DI    | Gap   | Status |
|--------|---------------|-----------------|-------|-------|--------|
| T0     | 43.4%         | 73.1%           | 0.594 | 0.297 | 🚨 ALERT |
| T1     | 27.6%         | 75.1%           | 0.368 | 0.475 | 🚨 ALERT |
| T2     | 27.6%         | 73.5%           | 0.376 | 0.458 | 🚨 ALERT |
| T3     | 28.2%         | 73.7%           | 0.383 | 0.455 | 🚨 ALERT |

#### Time-Series — USA Income Segment (Low vs Medium)

| Period | Low (<$40K) | Medium (40-80K) | DI    | Gap   | Status |
|--------|------------|-----------------|-------|-------|--------|
| T0     | 51.0%      | 59.5%           | 0.856 | 0.086 | ⚠️ WARNING |
| T1     | 28.0%      | 60.8%           | 0.460 | 0.328 | 🚨 ALERT |
| T2     | 27.8%      | 58.5%           | 0.475 | 0.307 | 🚨 ALERT |
| T3     | 28.5%      | 59.5%           | 0.478 | 0.311 | 🚨 ALERT |

#### Time-Series — USA Product Type (Credit Card vs Personal Loan)

| Period | Credit Card | Personal Loan | DI    | Gap   | Status |
|--------|------------|---------------|-------|-------|--------|
| T0     | 62.1%      | 51.2%         | 1.212 | 0.109 | ✅ FAIR |
| T1     | 48.2%      | 50.6%         | 0.953 | 0.024 | ✅ OK  |
| T2     | 47.1%      | 50.6%         | 0.930 | 0.035 | ✅ OK  |
| T3     | 47.0%      | 51.4%         | 0.914 | 0.044 | ✅ OK  |

> Product type remains fair across all periods — a useful control result showing
> DriftSafe correctly distinguishes drifting from stable attributes.

#### Time-Series — USA Credit Score Group (Low vs Medium)

| Period | Low (300-600) | Medium (601-700) | DI    | Gap   | Status |
|--------|--------------|------------------|-------|-------|--------|
| T0     | 13.4%        | 64.9%            | 0.206 | 0.515 | 🚨 ALERT |
| T1     | 7.4%         | 64.9%            | 0.113 | 0.575 | 🚨 ALERT |
| T2     | 7.1%         | 64.3%            | 0.110 | 0.573 | 🚨 ALERT |
| T3     | 5.9%         | 65.2%            | 0.090 | 0.593 | 🚨 ALERT |

---

### Cross-Region Summary

| Attribute | India DI (T0→T3) | USA DI (T0→T3) | Verdict |
|-----------|-----------------|----------------|---------|
| Age Group | 0.867 → 0.575 | 0.594 → 0.383 | 🚨 Worsens in both |
| Income Segment | 1.136 → 0.573 | 0.856 → 0.478 | 🚨 Worsens in both |
| Product Type | — | 1.212 → 0.914 | ✅ Stable |
| Credit Score | — | 0.206 → 0.090 | 🚨 Severe in USA |

---

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn flask openpyxl

# Instant fairness check (India, no server)
python bias_quick_test.py

# Full India time-series pipeline
python india.py

# Full USA time-series pipeline
python usa_model.py

# Launch Flask dashboard (both regions)
python app.py
# Visit: http://localhost:5000
```

---

## 📁 Project Structure

```
driftsafe/
├── app.py                        # Flask backend — /india and /api/usa routes
├── india.py                      # India pipeline (4 attributes × 4 windows)
├── usa_model.py                  # USA pipeline + get_fairness_metrics() Flask API
├── bias_quick_test.py            # Standalone quick eval (no server needed)
├── templates/
│   ├── dashboard.html            # Landing page — country selector
│   ├── india.html                # India fairness dashboard
│   └── usa.html                  # USA fairness dashboard
└── data/
    ├── IndianData_final.csv      # 4,269 Indian loan applications (2018–2024)
    └── Dated_Loan_Approval_Data.xlsx  # 50,000 US loan applications (2015–2025)
```

---

## 🔮 Future Work

- [ ] Statistical significance testing (bootstrap confidence intervals on DI)
- [ ] Counterfactual fairness analysis
- [ ] Automated retraining triggers when DI falls below threshold
- [ ] Additional metrics: Equalized Odds, Calibration, Predictive Parity
- [ ] Real-time streaming ingestion for live deployment monitoring
- [ ] Export fairness reports to PDF
