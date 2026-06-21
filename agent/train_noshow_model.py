"""
train_noshow_model.py

Trains a logistic regression model (via scikit-learn -- the Python
machine learning library) to predict no-show probability per appointment.

Key fix vs first attempt: class_weight='balanced' tells scikit-learn to
automatically weight the minority class (No Show, ~18%) more heavily
during training, so the model actually learns to predict no-shows rather
than always defaulting to the majority class (Kept).
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib

print("Connecting to Postgres and pulling appointment data...")

conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="caboodle_access", user="postgres"
)

query = """
    SELECT
        a.appointment_key,
        a.patient_key,
        a.lead_time_days,
        a.appointment_type,
        a.is_no_show,
        d.day_of_week,
        d.is_weekend,
        COALESCE(hist.prior_noshow_count, 0) AS prior_noshow_count,
        COALESCE(hist.prior_appt_count, 0)   AS prior_appt_count
    FROM analytics_marts.fact_appointments a
    LEFT JOIN analytics_staging.stg_dates d
        ON d.calendar_date = a.scheduled_datetime::date
    LEFT JOIN (
        SELECT
            a2.patient_key,
            a2.appointment_key,
            COUNT(*) FILTER (WHERE a1.is_no_show AND a1.scheduled_datetime < a2.scheduled_datetime)
                AS prior_noshow_count,
            COUNT(*) FILTER (WHERE a1.scheduled_datetime < a2.scheduled_datetime)
                AS prior_appt_count
        FROM analytics_marts.fact_appointments a2
        LEFT JOIN analytics_marts.fact_appointments a1
            ON a1.patient_key = a2.patient_key
        GROUP BY a2.patient_key, a2.appointment_key
    ) hist ON hist.appointment_key = a.appointment_key
"""

df = pd.read_sql(query, conn)
conn.close()
print(f"  Loaded {len(df)} appointment rows.")

df = pd.get_dummies(df, columns=["appointment_type", "day_of_week"], drop_first=True)

feature_cols = (
    ["lead_time_days", "is_weekend", "prior_noshow_count", "prior_appt_count"]
    + [c for c in df.columns if c.startswith("appointment_type_") or c.startswith("day_of_week_")]
)

X = df[feature_cols].fillna(0)
y = df["is_no_show"].astype(int)

print(f"  No-show rate in training data: {y.mean():.1%}")

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
    class_weight='balanced'   # penalizes no-show misses more heavily,
                               # forcing the model to actually learn the
                               # minority class instead of ignoring it.
)
model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"  AUC-ROC: {auc:.3f}")

print("\nClassification report (at 0.5 probability threshold):")
y_pred = (y_pred_proba >= 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=["Kept", "No Show"]))

joblib.dump(model, "agent/noshow_model.pkl")
joblib.dump(scaler, "agent/noshow_scaler.pkl")
joblib.dump(feature_cols, "agent/noshow_features.pkl")
print("\nModel saved to agent/noshow_model.pkl")
