"""
test_nightly_feed.py
Simulates the nightly feed directly, without needing Airflow.
Same logic as the DAG tasks, just called directly.
"""

import random
import subprocess
import psycopg2
import numpy as np
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "caboodle_access",
    "user": "postgres",
}

def generate_new_appointments():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(appointment_key) FROM raw.appointment_fact")
    max_key = cur.fetchone()[0] or 0
    cur.execute("SELECT patient_key FROM raw.patient_dim")
    patient_keys = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT department_key FROM raw.department_dim")
    department_keys = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT provider_key FROM raw.provider_dim")
    provider_keys = [r[0] for r in cur.fetchall()]

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    num_new = random.randint(5, 15)
    appointment_types = ["New Patient", "Follow-up", "Annual Wellness", "Procedure", "Therapy Session"]
    statuses = ["Completed", "No Show", "Cancelled", "Rescheduled"]
    status_weights = [0.73, 0.18, 0.06, 0.03]

    rows = []
    for i in range(num_new):
        lead_days = random.randint(1, 60)
        scheduled_dt = today + timedelta(days=lead_days,
                                          hours=random.randint(8, 16),
                                          minutes=random.choice([0, 15, 30, 45]))
        rows.append((
            f"APPT{max_key + i + 1:07d}",
            random.choice(patient_keys),
            random.choice(department_keys),
            random.choice(provider_keys),
            scheduled_dt,
            today,
            random.choices(statuses, weights=status_weights)[0],
            random.choice(appointment_types),
        ))

    cur.executemany(
        """INSERT INTO raw.appointment_fact
           (appointment_id, patient_key, department_key, provider_key,
            scheduled_datetime, booked_datetime, appointment_status, appointment_type)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        rows
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"  Inserted {num_new} new appointments.")


def generate_new_encounters():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT MAX(encounter_key) FROM raw.encounter_fact")
    max_key = cur.fetchone()[0] or 0
    cur.execute("SELECT patient_key FROM raw.patient_dim")
    patient_keys = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT department_key, department_type FROM raw.department_dim")
    dept_rows = cur.fetchall()
    all_dept_keys = [r[0] for r in dept_rows]
    inpatient_dept_keys = [r[0] for r in dept_rows if r[1] == "Inpatient Unit"]
    cur.execute("SELECT provider_key FROM raw.provider_dim")
    provider_keys = [r[0] for r in cur.fetchall()]

    today = datetime.now().replace(minute=0, second=0, microsecond=0)
    num_new = random.randint(2, 8)
    encounter_types = ["Inpatient", "Emergency", "Outpatient"]
    encounter_weights = [0.25, 0.35, 0.40]
    dispositions = ["Home", "Home Health", "SNF", "AMA", "Expired"]
    disposition_weights = [0.78, 0.10, 0.06, 0.04, 0.02]

    rows = []
    for i in range(num_new):
        encounter_type = random.choices(encounter_types, weights=encounter_weights)[0]
        if encounter_type == "Inpatient":
            dept_key = random.choice(inpatient_dept_keys)
            discharge_dt = today + timedelta(days=max(1, int(np.random.gamma(shape=1.5, scale=2.0))))
        elif encounter_type == "Emergency":
            dept_key = random.choice(all_dept_keys)
            discharge_dt = today + timedelta(hours=random.randint(1, 6))
        else:
            dept_key = random.choice(all_dept_keys)
            discharge_dt = today + timedelta(hours=1)

        rows.append((
            f"ENC{max_key + i + 1:07d}",
            random.choice(patient_keys),
            dept_key,
            random.choice(provider_keys),
            encounter_type,
            today,
            discharge_dt,
            random.choices(dispositions, weights=disposition_weights)[0],
        ))

    cur.executemany(
        """INSERT INTO raw.encounter_fact
           (encounter_id, patient_key, department_key, provider_key,
            encounter_type, admission_datetime, discharge_datetime, discharge_disposition)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        rows
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"  Inserted {num_new} new encounters.")


print("=== Simulating nightly feed ===")

print("\n[Task 1] Generating new appointments...")
generate_new_appointments()

print("\n[Task 2] Generating new encounters...")
generate_new_encounters()

print("\n[Task 3] Running dbt models...")
result = subprocess.run(
    ["dbt", "run"],
    cwd="/Users/sami/Desktop/DE Projects/caboodle_access_platform/dbt/caboodle_access",
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print("dbt error:", result.stderr)
else:
    print("dbt run completed successfully.")

print("\n=== Nightly feed complete ===")
