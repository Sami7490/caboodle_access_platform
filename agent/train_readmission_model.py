"""
train_readmission_model.py

Trains a logistic regression model (via scikit-learn -- the Python
machine learning library) to predict 30-day readmission probability
for inpatient encounters.

Features used (pulled from analytics_marts.fct_readmissions +
supporting tables):
  - length_of_stay_days    : longer stays often signal sicker patients
  - discharge_disposition  : SNF/AMA discharges have higher readmission risk
  - age_years              : patient age at time of admission
  - prior_admission_count  : how many prior inpatient stays this patient had
  - department_key         : which unit the patient was in
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("Connecting to Postgres and pulling inpatient encounter data...")

conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="caboodle_access", user="postgres"
)

query = """
    SELECT
        r.encounter_key,
        r.patient_key,
        r.length_of_stay_days,
        r.discharge_disposition,
        r.is_30_day_readmission,
        p.age_years,
        -- prior_admission_count: how many inpatient stays this patient
        -- had BEFORE this one (same data-leakage precaution as before --
        -- only look at admissions earlier in time, not later).
        COALESCE(hist.prior_admission_count, 0) AS prior_admission_count,
        r.department_key
    FROM analytics_marts.fct_readmissions r
    LEFT JOIN analytics_marts.dim_patients p
        ON p.patient_key = r.patient_key
    LEFT JOIN (
        SELECT
            r2.encounter_key,
            COUNT(*) FILTER (WHERE r1.admission_datetime < r2.admission_datetime)
                AS prior_admission_count
        FROM analytics_marts.fct_readmissions r2
        LEFT JOIN analytics_marts.fct_readmissions r1
            ON r1.patient_key = r2.patient_key
        GROUP BY r2.encounter_key
    ) hist ON hist.encounter_key = r.encounter_key
"""

df = pd.read_sql(query, conn)
conn.close()
print(f"  Loaded {len(df)} inpatient encounter rows.")
print(f"  Readmission rate in training data: {df['is_30_day_readmission'].mean():.1%}")

# One-hot encode discharge_disposition and department_key (categorical
# columns that need to become numeric 0/1 columns for scikit-learn).
df = pd.get_dummies(df, columns=["discharge_disposition", "department_key"], drop_first=True)

feature_cols = (
    ["length_of_stay_days", "age_years", "prior_admission_count"]
    + [c for c in df.columns if c.startswith("discharge_disposition_") or c.startswith("department_key_")]
)

X = df[feature_cols].fillna(0)
y = df["is_30_day_readmission"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nTraining logistic regression model (balanced class weights)...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"  AUC-ROC: {auc:.3f}")

print("\nClassification report (at 0.5 probability threshold):")
y_pred = (y_pred_proba >= 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=["Not Readmitted", "Readmitted"]))

joblib.dump(model, "agent/readmission_model.pkl")
joblib.dump(scaler, "agent/readmission_scaler.pkl")
joblib.dump(feature_cols, "agent/readmission_features.pkl")
print("\nModel saved to agent/readmission_model.pkl")
