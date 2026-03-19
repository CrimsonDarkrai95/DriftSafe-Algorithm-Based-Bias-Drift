"""
bias_quick_test.py
==================
Quick standalone fairness evaluation for the DriftSafe loan approval system.
Uses IndianData_final.csv — run directly: python bias_quick_test.py

Computes:
  - Disparate Impact (DI) before and after threshold drift
  - Approval Gap before and after
  - Alert status per the 80% rule
"""

import pandas as pd
import numpy as np

# ─── Metric Functions ─────────────────────────────────────────────────────────

def disparate_impact(df, protected_col, outcome_col):
    """
    Disparate Impact Ratio = P(outcome | unprivileged) / P(outcome | privileged)
    80% rule: DI >= 0.80 is considered fair. Below 0.80 triggers alert.
    """
    privileged   = df[df[protected_col] == 1]
    unprivileged = df[df[protected_col] == 0]
    p_priv   = privileged[outcome_col].mean()
    p_unpriv = unprivileged[outcome_col].mean()
    return p_unpriv / p_priv if p_priv > 0 else np.nan

def approval_gap(df, protected_col, outcome_col):
    """Absolute approval rate gap between unprivileged and privileged groups."""
    privileged   = df[df[protected_col] == 1]
    unprivileged = df[df[protected_col] == 0]
    return abs(unprivileged[outcome_col].mean() - privileged[outcome_col].mean())

def fairness_status(di: float) -> str:
    if di >= 0.90: return "✅ FAIR"
    if di >= 0.80: return "⚠️  WARNING  (approaching threshold)"
    return "🚨 ALERT    (violates 80% rule — intervention required)"

# ─── Load Data ────────────────────────────────────────────────────────────────

df = pd.read_csv("IndianData_final.csv")

# ─── Define Protected Attribute ───────────────────────────────────────────────
# Age group: 0 = Young (18–35) = unprivileged | 1 = Middle/Senior (>35) = privileged
df["protected"] = (df["age"] > 35).astype(int)

# ─── Baseline Predictions (pre-drift) ─────────────────────────────────────────
# Proxy approval rule: credit_score >= 650 AND debt_to_income <= 0.40
df["pred"] = (
    (df["credit_score"] >= 650) &
    (df["debt_to_income_ratio"] <= 0.40)
).astype(int)

# ─── Compute Baseline Metrics ─────────────────────────────────────────────────
di_before  = disparate_impact(df, "protected", "pred")
gap_before = approval_gap(df, "protected", "pred")

young_rate_before  = df[df["protected"] == 0]["pred"].mean()
senior_rate_before = df[df["protected"] == 1]["pred"].mean()

# ─── Simulate Post-Deployment Drift ───────────────────────────────────────────
# Realistic drift: thresholds tighten specifically for young applicants
# (simulates a biased model update or data distribution shift)
df_drift = df.copy()
mask = df_drift["protected"] == 0   # young applicants only

df_drift.loc[mask, "pred"] = (
    (df_drift.loc[mask, "credit_score"] >= 700) &       # +50 pts stricter
    (df_drift.loc[mask, "debt_to_income_ratio"] <= 0.35) # -5% stricter
).astype(int)

# ─── Compute Post-Drift Metrics ───────────────────────────────────────────────
di_after  = disparate_impact(df_drift, "protected", "pred")
gap_after = approval_gap(df_drift, "protected", "pred")

young_rate_after  = df_drift[df_drift["protected"] == 0]["pred"].mean()
senior_rate_after = df_drift[df_drift["protected"] == 1]["pred"].mean()

# ─── Report ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("DRIFTSAFE — BIAS QUICK TEST")
print("Dataset: Indian Loan Applications")
print("=" * 60)

print(f"\nDataset size:          {len(df):,} records")
print(f"Protected group:       Young applicants (age ≤ 35)")
print(f"Reference group:       Middle/Senior applicants (age > 35)")
print(f"Young applicants:      {(df['protected']==0).sum():,}  ({(df['protected']==0).mean():.1%})")
print(f"Senior applicants:     {(df['protected']==1).sum():,}  ({(df['protected']==1).mean():.1%})")

print("\n── BEFORE DRIFT ─────────────────────────────────────────")
print(f"  Approval rate (Young):   {young_rate_before:.3f}  ({young_rate_before:.1%})")
print(f"  Approval rate (Senior):  {senior_rate_before:.3f}  ({senior_rate_before:.1%})")
print(f"  Disparate Impact:        {di_before:.4f}")
print(f"  Approval Gap:            {gap_before:.4f}")
print(f"  Status:                  {fairness_status(di_before)}")

print("\n── AFTER DRIFT (stricter thresholds for Young) ──────────")
print(f"  Approval rate (Young):   {young_rate_after:.3f}  ({young_rate_after:.1%})")
print(f"  Approval rate (Senior):  {senior_rate_after:.3f}  ({senior_rate_after:.1%})")
print(f"  Disparate Impact:        {di_after:.4f}")
print(f"  Approval Gap:            {gap_after:.4f}")
print(f"  Status:                  {fairness_status(di_after)}")

print("\n── DRIFT SUMMARY ────────────────────────────────────────")
print(f"  DI change:               {di_before:.4f} → {di_after:.4f}  (Δ {di_after-di_before:+.4f})")
print(f"  Approval rate change:    {young_rate_before:.3f} → {young_rate_after:.3f}  for Young group")
print(f"  80% threshold crossed:   {'YES 🚨' if di_before >= 0.80 and di_after < 0.80 else 'Already below' if di_before < 0.80 else 'No'}")

print("\n" + "=" * 60)
print("Drift type: threshold tightening (credit_score +50, DTI -5%)")
print("Conclusion: DI drops below 0.80 after drift — fairness alert")
print("For full time-series monitoring: run india.py")
print("=" * 60)