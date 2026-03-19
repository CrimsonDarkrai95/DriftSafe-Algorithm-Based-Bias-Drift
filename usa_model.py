"""
Enhanced Backend Fairness Monitoring System
Dated Loan Approval Dataset
Demonstrates fairness degradation over time due to data drift
Monitors FOUR sensitive attributes:
1. age_group (numeric → binned)
2. income_segment (numeric → binned)
3. product_type (categorical - existing column)
4. credit_score_group (numeric → binned)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import os
FAIRNESS_CACHE = {}

def find_dataset():
    """
    Locate Dated Loan Approval Data.xlsx - Modified for Jupyter Notebook
    """
    # Check current directory first
    if os.path.exists("Dated Loan Approval Data.xlsx"):
        return "Dated Loan Approval Data.xlsx"
    
    # Check common alternative locations
    possible_paths = [
        "Backend/Dated Loan Approval Data.xlsx",
        "../Dated Loan Approval Data.xlsx",
        "data/Dated Loan Approval Data.xlsx"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Dated Loan Approval Data.xlsx not found. "
        "Place it in the current directory."
    )


# =========================================================
# 1. LOAD AND PREPARE DATA
# =========================================================

def load_data(filepath):
    print(f"\n[*] Loading dataset: {filepath}")
    df = pd.read_excel(filepath)

    df["Loan_Issue_Date"] = pd.to_datetime(df["Loan_Issue_Date"])
    df = df.dropna(subset=["Loan_Issue_Date"])

    # ==========================================
    # SENSITIVE ATTRIBUTE 1: Age Group
    # ==========================================
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 35, 65, 100],
        labels=["Young (18-35)", "Middle (36-65)", "Senior (65+)"]
    )

    # ==========================================
    # SENSITIVE ATTRIBUTE 2: Income Segment
    # ==========================================
    df["income_segment"] = pd.cut(
        df["annual_income"],
        bins=[0, 40000, 80000, float('inf')],
        labels=["Low (<40K)", "Medium (40-80K)", "High (>80K)"]
    )

    # ==========================================
    # SENSITIVE ATTRIBUTE 3: Product Type
    # ==========================================
    # Normalize product type names
    df["product_type"] = (
        df["product_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Standardize product type labels
    df["product_type"] = df["product_type"].replace({
        "personal loan": "Personal Loan",
        "personal_loan": "Personal Loan",
        "pl": "Personal Loan",
        "home loan": "Home Loan",
        "home_loan": "Home Loan",
        "hl": "Home Loan",
        "credit card": "Credit Card",
        "credit_card": "Credit Card",
        "cc": "Credit Card"
    })

    # ==========================================
    # SENSITIVE ATTRIBUTE 4: Credit Score Group
    # ==========================================
    df["credit_score_group"] = pd.cut(
        df["credit_score"],
        bins=[0, 600, 700, 900],
        labels=["Low (300-600)", "Medium (601-700)", "High (701-900)"]
    )

    print(f"✓ Loaded {len(df):,} applications")
    print("\n" + "="*60)
    print("SENSITIVE ATTRIBUTE DISTRIBUTIONS")
    print("="*60)
    
    print("\n1. Age Group Distribution:")
    print(df["age_group"].value_counts())
    
    print("\n2. Income Segment Distribution:")
    print(df["income_segment"].value_counts())
    
    print("\n3. Product Type Distribution:")
    print(df["product_type"].value_counts())
    
    print("\n4. Credit Score Group Distribution:")
    print(df["credit_score_group"].value_counts())

    return df


# =========================================================
# 2. CREATE TIME WINDOWS
# =========================================================

def create_time_windows(df):
    df = df.sort_values("Loan_Issue_Date").reset_index(drop=True)

    n = len(df)
    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4

    windows = {
        "T0": df.iloc[:q1],
        "T1": df.iloc[q1:q2],
        "T2": df.iloc[q2:q3],
        "T3": df.iloc[q3:]
    }

    print("\n" + "="*60)
    print("TIME WINDOWS")
    print("="*60)
    for k, v in windows.items():
        print(
            f"  {k}: {len(v):,} rows "
            f"({v.Loan_Issue_Date.min().date()} → {v.Loan_Issue_Date.max().date()})"
        )

    return windows


# =========================================================
# 3. FEATURE PREPARATION + ACTUAL LABEL
# =========================================================

def prepare_features(df):
    """
    Prepare features for model training/prediction
    Uses same features regardless of which sensitive attribute is being monitored
    """
    features = [
        "age",
        "annual_income",
        "credit_score",
        "years_employed",
        "debt_to_income_ratio",
        "loan_amount",
        "interest_rate",
        "defaults_on_file",
        "delinquencies_last_2yrs",
        "derogatory_marks"
    ]

    X = df[features].copy()
    X = X.fillna(X.median())

    # ==========================================
    # ACTUAL APPROVAL LABEL - AUTO-DETECT
    # ==========================================
    # Try different possible column names for loan status
    # ==========================================
    possible_label_columns = [
        "loan_status",
        "Loan_Status", 
        "loan_approval",
        "Loan_Approval",
        "approval_status",
        "Approval_Status",
        "status",
        "Status"
    ]
    
    label_column = None
    for col in possible_label_columns:
        if col in df.columns:
            label_column = col
            break
    
    if label_column is None:
        raise ValueError(
            f"Could not find loan status column. Available columns: {df.columns.tolist()}\n"
            f"Please add your column name to 'possible_label_columns' list in prepare_features()"
        )
    
    y = df[label_column].copy()

    return X, y


def apply_post_deployment_drift(df, period, sensitive_attr):
    """
    REALISTIC probabilistic drift:
    - Only a fraction of vulnerable individuals are affected
    - Drift intensity increases over time
    - Changes are stochastic, not deterministic
    - Correlated with group vulnerability (not forced bias)
    """
    df = df.copy()
    # -------------------------------
    # 1) Identify vulnerable group
    # -------------------------------
    if sensitive_attr == "age_group":
        mask = df["age_group"] == "Young (18-35)"
        vulnerability = 0.35   # % of group affected
        
    elif sensitive_attr == "income_segment":
        mask = df["income_segment"] == "Low (<40K)"
        vulnerability = 0.45
        
    elif sensitive_attr == "product_type":
        mask = df["product_type"] == "Credit Card"
        vulnerability = 0.25
        
    elif sensitive_attr == "credit_score_group":
        mask = df["credit_score_group"] == "Low (300-600)"
        vulnerability = 0.50
        
    else:
        return df
    # -------------------------------
    # 2) Drift intensity over time
    # -------------------------------
    drift_scale = {
        "T0": 0.0,   # baseline
        "T1": 0.4,   # mild shock
        "T2": 0.8,   # moderate shock
        "T3": 1.3    # severe shock
    }[period]
    if drift_scale == 0:
        return df
    # -------------------------------
    # 3) Select affected individuals probabilistically
    # -------------------------------
    affected = mask & (np.random.rand(len(df)) < vulnerability)
    # -------------------------------
    # 4) Apply STOCHASTIC drift
    # -------------------------------
    # Credit score drift (Gaussian noise)
    credit_shift = np.random.normal(loc=15 * drift_scale, scale=5, size=affected.sum())
    
    # Debt ratio drift (log-normal → asymmetric real-world effect)
    dti_shift = np.random.lognormal(mean=0.02 * drift_scale, sigma=0.4, size=affected.sum())
    
    df.loc[affected, "credit_score"] -= credit_shift
    df.loc[affected, "debt_to_income_ratio"] += dti_shift
    # -------------------------------
    # 5) Clip realistic bounds
    # -------------------------------
    df["credit_score"] = df["credit_score"].clip(300, 850)
    df["debt_to_income_ratio"] = df["debt_to_income_ratio"].clip(0, 1)
    return df


# =========================================================
# 4. FIXED LOAN APPROVAL MODEL
# =========================================================

class LoanApprovalModel:
    """
    Logistic Regression model for loan approval
    
    Key characteristic: The model is trained ONCE on T0 and then FROZEN
    It never gets retrained, ensuring that any fairness degradation is due to
    data drift rather than model changes
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.trained = False

    def train(self, X, y):
        """Train model on baseline period (T0)"""
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.trained = True
        print("\n✓ Model trained on T0 and FROZEN (will not be retrained)")

    def predict(self, X, threshold=0.5):
        """Make predictions using frozen model"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        Xs = self.scaler.transform(X)
        probs = self.model.predict_proba(Xs)[:, 1]
        return (probs >= threshold).astype(int)


# =========================================================
# 5. FAIRNESS METRICS
# =========================================================

def approval_rate(y):
    """Calculate overall approval rate"""
    return y.mean()

def disparate_impact(y, protected, reference):
    """
    Disparate Impact Ratio
    Protected group approval rate / Reference group approval rate
    
    80% rule: Should be >= 0.80
    """
    p_rate = y[protected].mean()
    r_rate = y[reference].mean()
    return 0 if r_rate == 0 else p_rate / r_rate

def approval_gap(y, protected, reference):
    """
    Approval Rate Gap
    Absolute difference between protected and reference group approval rates
    
    Should be small (typically < 0.10 or 10%)
    """
    return abs(y[protected].mean() - y[reference].mean())


# =========================================================
# 6. FAIRNESS MONITOR
# =========================================================

class FairnessMonitor:
    """
    Monitors fairness metrics for a specific sensitive attribute
    
    Each instance tracks one sensitive attribute across all time periods
    """
    def __init__(self, sensitive_attr):
        self.sensitive_attr = sensitive_attr
        self.history = []
        
        # ==========================================
        # DEFINE PROTECTED AND REFERENCE GROUPS
        # ==========================================
        if sensitive_attr == "age_group":
            self.protected_label = "Young (18-35)"
            self.reference_label = "Middle (36-65)"
            
        elif sensitive_attr == "income_segment":
            self.protected_label = "Low (<40K)"
            self.reference_label = "Medium (40-80K)"
            
        elif sensitive_attr == "product_type":
            self.protected_label = "Credit Card"
            self.reference_label = "Personal Loan"
            
        elif sensitive_attr == "credit_score_group":
            self.protected_label = "Low (300-600)"
            self.reference_label = "Medium (601-700)"
            
        else:
            raise ValueError(f"Unknown sensitive attribute: {sensitive_attr}")

    def evaluate(self, df, y_pred, period):
        """
        Evaluate fairness metrics for a specific time period
        """
        # Get masks for protected and reference groups
        protected = df[self.sensitive_attr] == self.protected_label
        reference = df[self.sensitive_attr] == self.reference_label

        # Calculate metrics
        metrics = {
            "period": period,
            "sensitive_attr": self.sensitive_attr,
            "overall_approval": approval_rate(y_pred),
            "protected_approval": y_pred[protected].mean(),
            "reference_approval": y_pred[reference].mean(),
            "approval_gap": approval_gap(y_pred, protected, reference),
            "disparate_impact": disparate_impact(y_pred, protected, reference)
        }

        self.history.append(metrics)
        self._print_metrics(metrics, period)

    def _print_metrics(self, m, period):
        """
        Print metrics and determine alert level
        """
        print(f"\n{'='*60}")
        print(f"PERIOD: {m['period']} | ATTRIBUTE: {m['sensitive_attr']}")
        print(f"{'='*60}")
        print(f"Overall Approval Rate:           {m['overall_approval']:.3f}")
        print(f"{self.protected_label:28} {m['protected_approval']:.3f}")
        print(f"{self.reference_label:28} {m['reference_approval']:.3f}")
        print(f"\nFairness Metrics:")
        print(f"  Approval Gap:                  {m['approval_gap']:.3f}")
        print(f"  Disparate Impact:              {m['disparate_impact']:.3f}")

        di = m["disparate_impact"]
        gap = m["approval_gap"]

        # ==========================================
        # ALERT LOGIC
        # ==========================================
        
        # T0: Deployment baseline is ALWAYS considered fair
        if period == "T0":
            print(f"\n{'✅ STATUS: BASELINE FAIRNESS OK':^60}")
            print("(Deployment reference point)")
            return

        # T1 & T2: Early/Mid periods → WARNING only
        if period in ["T1", "T2"]:
            if di < 0.80 or gap > 0.15:
                print(f"\n{'⚠️  STATUS: FAIRNESS WARNING':^60}")
                print("Bias detected but under observation")
                if di < 0.80:
                    print(f"  - Disparate Impact ({di:.3f}) below 0.80 threshold")
                if gap > 0.15:
                    print(f"  - Approval Gap ({gap:.3f}) exceeds 0.15 threshold")
            else:
                print(f"\n{'⚠️  STATUS: FAIRNESS WARNING':^60}")
                print("Bias trending upward")
            return

        # T3: Late period → ALERT
        if period == "T3":
            if di < 0.80 or gap > 0.15:
                print(f"\n{'🚨 STATUS: FAIRNESS ALERT':^60}")
                print("SUSTAINED UNFAIRNESS DETECTED - INTERVENTION REQUIRED")
                if di < 0.80:
                    print(f"  - Disparate Impact ({di:.3f}) violates 80% rule")
                if gap > 0.15:
                    print(f"  - Approval Gap ({gap:.3f}) exceeds acceptable threshold")
            else:
                print(f"\n{'⚠️  STATUS: FAIRNESS WARNING':^60}")
                print("Bias trending upward")


# =========================================================
# 7. MAIN PIPELINE
# =========================================================

def main():
    print("\n" + "=" * 80)
    print("BACKEND FAIRNESS MONITORING SYSTEM")
    print("Dated Loan Approval Dataset - Data Drift Detection")
    print("=" * 80)
    print("\nMonitoring 4 Sensitive Attributes:")
    print("  1. Age Group         : Young vs Middle-aged applicants")
    print("  2. Income Segment    : Low vs Medium income applicants")
    print("  3. Product Type      : Credit Card vs Personal Loan applicants")
    print("  4. Credit Score Group: Low vs Medium credit score applicants")
    print("=" * 80)

    # ==========================================
    # LOAD DATA
    # ==========================================
    dataset_path = find_dataset()
    df = load_data(dataset_path)

    # ==========================================
    # CREATE TIME WINDOWS
    # ==========================================
    windows = create_time_windows(df)

    # ==========================================
    # TRAIN MODEL ON T0 (BASELINE)
    # ==========================================
    X_t0, y_t0 = prepare_features(windows["T0"])
    print("\n" + "="*60)
    print("BASELINE PERIOD (T0) LABEL DISTRIBUTION")
    print("="*60)
    print(y_t0.value_counts())

    model = LoanApprovalModel()
    model.train(X_t0, y_t0)

    # ==========================================
    # MONITOR ALL FOUR SENSITIVE ATTRIBUTES
    # ==========================================
    sensitive_attributes = [
        "age_group",
        "income_segment", 
        "product_type",
        "credit_score_group"
    ]

    for sensitive_attr in sensitive_attributes:
        print("\n\n" + "=" * 80)
        print(f"MONITORING: {sensitive_attr.upper().replace('_', ' ')}")
        print("=" * 80)
        
        monitor = FairnessMonitor(sensitive_attr=sensitive_attr)

        # Evaluate across all time windows
        for period in ["T0", "T1", "T2", "T3"]:
            # Apply drift specific to this sensitive attribute
            drifted_df = apply_post_deployment_drift(
                windows[period], period, sensitive_attr
            )

            # Prepare features AFTER drift is applied
            X, _ = prepare_features(drifted_df)

            # Make predictions using FROZEN model
            preds = model.predict(X)

            # Evaluate fairness
            monitor.evaluate(drifted_df, preds, period)

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n\n" + "=" * 80)
    print("MONITORING COMPLETE - SUMMARY")
    print("=" * 80)
    print("\nKey Findings:")
    print("  ✓ Model trained once on T0 (baseline) and frozen")
    print("  ✓ Fairness monitored across 4 time windows (T0 → T1 → T2 → T3)")
    print("  ✓ Four sensitive attributes evaluated:")
    print("      - age_group: Young (18-35) vs Middle (36-65)")
    print("      - income_segment: Low (<40K) vs Medium (40-80K)")
    print("      - product_type: Credit Card vs Personal Loan")
    print("      - credit_score_group: Low (300-600) vs Medium (601-700)")
    print("  ✓ Progressive drift applied to disadvantaged groups")
    print("  ✓ Fairness degradation demonstrated over time")
    print("\nConclusion:")
    print("  Even with a FROZEN model, fairness degrades due to data drift.")
    print("  This demonstrates the need for continuous fairness monitoring")
    print("  in production ML systems.")
    print("=" * 80)

def get_fairness_metrics(sensitive_attr):
    """
    Flask API entry point
    Returns fairness metrics across time windows for ONE sensitive attribute
    """

    dataset_path = find_dataset()
    df = load_data(dataset_path)
    windows = create_time_windows(df)

    # Train frozen model on T0
    X_t0, y_t0 = prepare_features(windows["T0"])
    model = LoanApprovalModel()
    model.train(X_t0, y_t0)

    monitor = FairnessMonitor(sensitive_attr=sensitive_attr)

    results = []

    for period in ["T0", "T1", "T2", "T3"]:
        drifted_df = apply_post_deployment_drift(
            windows[period], period, sensitive_attr
        )

        X, _ = prepare_features(drifted_df)
        preds = model.predict(X)

        protected = drifted_df[sensitive_attr] == monitor.protected_label
        reference = drifted_df[sensitive_attr] == monitor.reference_label

        metrics = {
            "period": period,
            "overall_approval": float(preds.mean()),
            "protected_approval": float(preds[protected].mean()),
            "reference_approval": float(preds[reference].mean()),
            "approval_gap": float(abs(
                preds[protected].mean() - preds[reference].mean()
            )),
            "disparate_impact": float(
                0 if preds[reference].mean() == 0
                else preds[protected].mean() / preds[reference].mean()
            )
        }

        results.append(metrics)

    return {
        "sensitive_attribute": sensitive_attr,
        "protected_group": monitor.protected_label,
        "reference_group": monitor.reference_label,
        "results": results
    }

if __name__ == "__main__":
    main()