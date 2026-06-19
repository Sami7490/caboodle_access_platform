"""
generate_synthetic_data.py

Generates synthetic (fake) patient, encounter, and appointment data and
loads it into our local Postgres database.

Tools used in this script:
  - Faker     (generates realistic-looking fake names, birthdates, etc.)
  - numpy     (does weighted random sampling, so events aren't purely
               uniform random -- this is how we bake in realistic risk
               patterns like "some patients chronically no-show")
  - psycopg2  (the Python library that lets us talk to Postgres directly)

Design note: each synthetic patient gets a hidden "risk profile" that
influences the PROBABILITY of certain events (no-shows, readmissions).
We do NOT store this hidden risk profile as a column in the database --
that would be cheating, since in real life you never have a literal
"true risk score" column. Only the resulting OBSERVABLE events (the
actual no-shows, the actual readmissions) get written to the database,
which is exactly what a real predictive model would have to learn from.
"""

import argparse
import random
import psycopg2
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

NUM_PATIENTS = 500
YEARS_OF_HISTORY = 3
TODAY = datetime.now()  # use the real current date/time as our reference point
START_DATE = TODAY - timedelta(days=365 * YEARS_OF_HISTORY)

# Postgres connection settings. Password matches the postgres role and the
# dbt profile (~/.dbt/profiles.yml).
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "caboodle_access",
    "user": "postgres",
    "password": "epic",
}

# Faker (to generate realistic fake names/demographics), seeded so the
# dataset is reproducible if you re-run the script.
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------------
# REFERENCE DATA: departments and providers (hand-curated, not random, so
# the dashboard has recognizable, realistic department names).
# ----------------------------------------------------------------------------

DEPARTMENTS = [
    # (department_name, department_type, specialty)
    ("General Pediatrics Clinic", "Outpatient Clinic", "General Pediatrics"),
    ("Pediatric Cardiology", "Outpatient Clinic", "Cardiology"),
    ("Pediatric Endocrinology", "Outpatient Clinic", "Endocrinology"),
    ("Pediatric Pulmonology", "Outpatient Clinic", "Pulmonology"),
    ("Pediatric Emergency Department", "Emergency Department", "Emergency Medicine"),
    ("Pediatric Inpatient Unit", "Inpatient Unit", "General Pediatrics"),
    ("Pediatric ICU", "Inpatient Unit", "Critical Care"),
    ("Behavioral Health Clinic", "Outpatient Clinic", "Psychiatry"),
]

PROVIDER_TYPES = ["MD", "NP", "PA"]


def generate_providers(n=20):
    """Generate a pool of fake providers tied to specialties matching our
    department list, so e.g. a Cardiology department is staffed by
    providers whose specialty is Cardiology."""
    specialties = list({d[2] for d in DEPARTMENTS})
    providers = []
    for _ in range(n):
        specialty = random.choice(specialties)
        provider_type = random.choice(PROVIDER_TYPES)
        name = f"Dr. {fake.last_name()}" if provider_type == "MD" else fake.last_name()
        providers.append((name, provider_type, specialty))
    return providers


# ----------------------------------------------------------------------------
# PATIENT RISK PROFILES (hidden -- never written to the database)
# ----------------------------------------------------------------------------

def generate_patient_risk_profile():
    """
    Returns a dict of hidden risk parameters for one patient.

    no_show_tendency: baseline probability this patient no-shows any given
        scheduled appointment. Drawn from a Beta distribution so most
        patients cluster low, with a realistic tail of higher-risk
        patients (rather than every patient being equally likely to be
        high-risk, which a flat/uniform distribution would do).

    readmission_tendency: baseline probability that, given an inpatient
        admission, this patient gets readmitted within 30 days.
    """
    no_show_tendency = np.random.beta(a=2, b=13)        # population mean ~13%
    readmission_tendency = np.random.beta(a=1.5, b=14)  # population mean ~10%
    return {
        "no_show_tendency": no_show_tendency,
        "readmission_tendency": readmission_tendency,
    }


# ----------------------------------------------------------------------------
# PATIENT GENERATION
# ----------------------------------------------------------------------------

def generate_patients(n):
    """Generate n synthetic patients (pediatric population, 0-17 years old
    as of the start of our history window)."""
    patients = []
    risk_profiles = []
    for i in range(n):
        dob = fake.date_of_birth(minimum_age=0, maximum_age=17)
        mrn = f"MRN{100000 + i}"
        patients.append({
            "mrn": mrn,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "date_of_birth": dob,
            "sex": random.choice(["Male", "Female"]),
            "race": random.choice([
                "White", "Black or African American", "Asian",
                "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander",
                "Other", "Unknown"
            ]),
            "ethnicity": random.choice(["Hispanic or Latino", "Not Hispanic or Latino", "Unknown"]),
            "zip_code": fake.zipcode(),
            "primary_language": random.choices(
                ["English", "Spanish", "Other"], weights=[0.82, 0.12, 0.06]
            )[0],
        })
        risk_profiles.append(generate_patient_risk_profile())
    return patients, risk_profiles


# ----------------------------------------------------------------------------
# APPOINTMENT GENERATION
# ----------------------------------------------------------------------------

APPOINTMENT_TYPES = ["New Patient", "Follow-up", "Annual Wellness", "Procedure", "Therapy Session"]


def generate_appointments_for_patient(patient_key, risk_profile, department_keys, provider_keys):
    """Generate a realistic series of appointments for one patient across
    the full history window, including a lead-time effect on no-show risk
    (booking further in advance increases no-show probability)."""
    appointments = []
    num_appointments = np.random.poisson(lam=6) + 1  # at least 1 appointment per patient

    for _ in range(num_appointments):
        days_offset = random.randint(0, (TODAY - START_DATE).days)
        scheduled_dt = START_DATE + timedelta(days=days_offset)
        scheduled_dt = scheduled_dt.replace(
            hour=random.randint(8, 16), minute=random.choice([0, 15, 30, 45])
        )

        # Lead time: gamma distribution -- most appointments booked a few
        # days to a few weeks out, with a tail of longer lead times.
        lead_time_days = max(1, int(np.random.gamma(shape=2.0, scale=10.0)))
        booked_dt = scheduled_dt - timedelta(days=lead_time_days)

        # Tuned so the OVERALL no-show rate lands around 15-18%, a
        # realistic range for pediatric outpatient settings.
        lead_time_effect = min(0.12, lead_time_days / 500)
        no_show_prob = min(0.9, risk_profile["no_show_tendency"] + lead_time_effect)

        roll = random.random()
        if roll < no_show_prob:
            status = "No Show"
        elif roll < no_show_prob + 0.05:
            status = "Cancelled"
        elif roll < no_show_prob + 0.08:
            status = "Rescheduled"
        else:
            status = "Completed"

        appointments.append({
            "patient_key": patient_key,
            "department_key": random.choice(department_keys),
            "provider_key": random.choice(provider_keys),
            "scheduled_datetime": scheduled_dt,
            "booked_datetime": booked_dt,
            "appointment_status": status,
            "appointment_type": random.choice(APPOINTMENT_TYPES),
        })

    return appointments


# ----------------------------------------------------------------------------
# ENCOUNTER GENERATION (inpatient / ED / outpatient visits + readmissions)
# ----------------------------------------------------------------------------

DISCHARGE_DISPOSITIONS = ["Home", "Home Health", "SNF", "AMA", "Expired"]
DISPOSITION_WEIGHTS = [0.78, 0.10, 0.06, 0.04, 0.02]  # "Home" dominates, "Expired" rare


def generate_encounters_for_patient(patient_key, risk_profile, department_keys, provider_keys,
                                     inpatient_dept_keys):
    """
    Generate a realistic encounter history for one patient, including
    chains of 30-day READMISSIONS for higher-risk patients.

    Strategy: generate independent inpatient "episodes." Based on the
    patient's readmission_tendency, we probabilistically chain on a second
    (and possibly third+) inpatient admission within 30 days of the prior
    discharge -- a true readmission. This creates realistic clustering:
    readmissions aren't independent random events, they cluster within
    certain higher-risk patients, which is exactly what a readmission
    model needs to learn from.
    """
    encounters = []
    num_episodes = np.random.poisson(lam=0.6)  # most patients: 0, some: 1-2+

    for _ in range(num_episodes):
        days_offset = random.randint(0, (TODAY - START_DATE).days - 40)
        admit_dt = START_DATE + timedelta(days=days_offset)
        admit_dt = admit_dt.replace(hour=random.randint(0, 23), minute=0)

        los_days = max(1, int(np.random.gamma(shape=1.5, scale=2.0)))
        discharge_dt = admit_dt + timedelta(days=los_days)
        disposition = random.choices(DISCHARGE_DISPOSITIONS, weights=DISPOSITION_WEIGHTS)[0]

        encounters.append({
            "patient_key": patient_key,
            "department_key": random.choice(inpatient_dept_keys),
            "provider_key": random.choice(provider_keys),
            "encounter_type": "Inpatient",
            "admission_datetime": admit_dt,
            "discharge_datetime": discharge_dt,
            "discharge_disposition": disposition,
        })

        # Chain on readmissions based on this patient's hidden tendency.
        chain_date = discharge_dt
        while random.random() < risk_profile["readmission_tendency"]:
            gap_days = random.randint(2, 30)  # readmitted within 30 days
            readmit_dt = chain_date + timedelta(days=gap_days)
            if readmit_dt > TODAY:
                break
            los_days = max(1, int(np.random.gamma(shape=1.5, scale=2.0)))
            readmit_discharge_dt = readmit_dt + timedelta(days=los_days)
            disposition = random.choices(DISCHARGE_DISPOSITIONS, weights=DISPOSITION_WEIGHTS)[0]

            encounters.append({
                "patient_key": patient_key,
                "department_key": random.choice(inpatient_dept_keys),
                "provider_key": random.choice(provider_keys),
                "encounter_type": "Inpatient",
                "admission_datetime": readmit_dt,
                "discharge_datetime": readmit_discharge_dt,
                "discharge_disposition": disposition,
            })
            chain_date = readmit_discharge_dt

    # Separate, non-chained ED visits and outpatient encounters -- e.g. an
    # ED visit for a sprained ankle isn't part of readmission logic.
    num_other_encounters = np.random.poisson(lam=2)
    for _ in range(num_other_encounters):
        days_offset = random.randint(0, (TODAY - START_DATE).days)
        admit_dt = START_DATE + timedelta(days=days_offset)
        encounter_type = random.choice(["Emergency", "Outpatient"])
        los_hours = random.randint(1, 6) if encounter_type == "Emergency" else 1
        discharge_dt = admit_dt + timedelta(hours=los_hours)

        dept_pool = [d for d in department_keys if d not in inpatient_dept_keys] \
            if encounter_type == "Outpatient" else department_keys

        encounters.append({
            "patient_key": patient_key,
            "department_key": random.choice(dept_pool) if dept_pool else random.choice(department_keys),
            "provider_key": random.choice(provider_keys),
            "encounter_type": encounter_type,
            "admission_datetime": admit_dt,
            "discharge_datetime": discharge_dt,
            "discharge_disposition": "Home",
        })

    return encounters


# ----------------------------------------------------------------------------
# DATE DIMENSION GENERATION
# ----------------------------------------------------------------------------

def generate_date_dim(start_date, end_date):
    """One row per calendar day, with standard rollup attributes."""
    rows = []
    current = start_date
    while current <= end_date:
        date_key = int(current.strftime("%Y%m%d"))
        rows.append((
            date_key,
            current.date(),
            current.strftime("%A"),
            current.isoweekday(),
            current.weekday() >= 5,
            current.month,
            current.strftime("%B"),
            (current.month - 1) // 3 + 1,
            current.year,
        ))
        current += timedelta(days=1)
    return rows


# ----------------------------------------------------------------------------
# DATABASE LOADING
# ----------------------------------------------------------------------------

def load_all_data(reset=False):
    """Main entry point: generates all synthetic data and loads it into
    Postgres using psycopg2.

    If reset=True, TRUNCATEs the raw tables before loading so the script can
    be re-run cleanly. If reset=False (default) and the tables already hold
    data, aborts before inserting anything rather than tripping a unique
    constraint mid-run."""

    print("Connecting to Postgres...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Tables listed children-first so the TRUNCATE order is safe for the FK
    # constraints (appointment/encounter facts reference the dim tables).
    # CASCADE + RESTART IDENTITY also resets the SERIAL key sequences, so a
    # fresh load starts numbering from 1 again. This runs inside the same
    # transaction as the inserts below, so any later failure rolls it back too.
    if reset:
        print("--reset: truncating raw tables before load...")
        cur.execute(
            """TRUNCATE raw.appointment_fact, raw.encounter_fact, raw.patient_dim,
                       raw.provider_dim, raw.department_dim, raw.date_dim
               RESTART IDENTITY CASCADE"""
        )
    else:
        # Insert-only mode: fail fast (and cleanly) if data already exists,
        # instead of appending duplicate reference rows and then dying on the
        # patient MRN unique constraint partway through.
        cur.execute("SELECT count(*) FROM raw.patient_dim")
        existing = cur.fetchone()[0]
        if existing > 0:
            conn.rollback()
            cur.close()
            conn.close()
            raise SystemExit(
                f"Aborting: raw.patient_dim already contains {existing} rows. "
                "This script is insert-only and would create duplicates. "
                "Re-run with --reset to TRUNCATE the raw tables first."
            )

    print("Generating date_dim...")
    date_rows = generate_date_dim(START_DATE, TODAY + timedelta(days=90))  # pad 90 days for future scheduling
    cur.executemany(
        """INSERT INTO raw.date_dim
           (date_key, calendar_date, day_of_week, day_of_week_num, is_weekend,
            month_num, month_name, quarter_num, year_num)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (date_key) DO NOTHING""",
        date_rows
    )
    print(f"  Inserted {len(date_rows)} date_dim rows.")

    print("Generating department_dim...")
    cur.executemany(
        "INSERT INTO raw.department_dim (department_name, department_type, specialty) VALUES (%s,%s,%s)",
        DEPARTMENTS
    )
    cur.execute("SELECT department_key, department_name, department_type FROM raw.department_dim")
    dept_rows = cur.fetchall()
    department_keys = [r[0] for r in dept_rows]
    inpatient_dept_keys = [r[0] for r in dept_rows if r[2] == "Inpatient Unit"]
    print(f"  Inserted {len(dept_rows)} departments ({len(inpatient_dept_keys)} inpatient).")

    print("Generating provider_dim...")
    providers = generate_providers(n=20)
    cur.executemany(
        "INSERT INTO raw.provider_dim (provider_name, provider_type, specialty) VALUES (%s,%s,%s)",
        providers
    )
    cur.execute("SELECT provider_key FROM raw.provider_dim")
    provider_keys = [r[0] for r in cur.fetchall()]
    print(f"  Inserted {len(provider_keys)} providers.")

    print(f"Generating {NUM_PATIENTS} patients...")
    patients, risk_profiles = generate_patients(NUM_PATIENTS)
    patient_rows = [
        (p["mrn"], p["first_name"], p["last_name"], p["date_of_birth"], p["sex"],
         p["race"], p["ethnicity"], p["zip_code"], p["primary_language"])
        for p in patients
    ]
    cur.executemany(
        """INSERT INTO raw.patient_dim
           (mrn, first_name, last_name, date_of_birth, sex, race, ethnicity, zip_code, primary_language)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        patient_rows
    )
    cur.execute("SELECT patient_key FROM raw.patient_dim ORDER BY patient_key")
    patient_keys = [r[0] for r in cur.fetchall()]
    print(f"  Inserted {len(patient_keys)} patients.")

    print("Generating appointments for each patient...")
    all_appointments = []
    for idx, patient_key in enumerate(patient_keys):
        appts = generate_appointments_for_patient(patient_key, risk_profiles[idx], department_keys, provider_keys)
        all_appointments.extend(appts)

    appt_rows = [
        (f"APPT{i+1:07d}", a["patient_key"], a["department_key"], a["provider_key"],
         a["scheduled_datetime"], a["booked_datetime"], a["appointment_status"], a["appointment_type"])
        for i, a in enumerate(all_appointments)
    ]
    cur.executemany(
        """INSERT INTO raw.appointment_fact
           (appointment_id, patient_key, department_key, provider_key,
            scheduled_datetime, booked_datetime, appointment_status, appointment_type)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        appt_rows
    )
    print(f"  Inserted {len(appt_rows)} appointments.")

    print("Generating encounters (including readmission chains) for each patient...")
    all_encounters = []
    for idx, patient_key in enumerate(patient_keys):
        encs = generate_encounters_for_patient(patient_key, risk_profiles[idx], department_keys,
                                                 provider_keys, inpatient_dept_keys)
        all_encounters.extend(encs)

    enc_rows = [
        (f"ENC{i+1:07d}", e["patient_key"], e["department_key"], e["provider_key"],
         e["encounter_type"], e["admission_datetime"], e["discharge_datetime"], e["discharge_disposition"])
        for i, e in enumerate(all_encounters)
    ]
    cur.executemany(
        """INSERT INTO raw.encounter_fact
           (encounter_id, patient_key, department_key, provider_key, encounter_type,
            admission_datetime, discharge_datetime, discharge_disposition)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        enc_rows
    )
    print(f"  Inserted {len(enc_rows)} encounters.")

    conn.commit()
    cur.close()
    conn.close()
    print("Done. All synthetic data loaded into Postgres.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Caboodle access/utilization data and load it into Postgres."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="TRUNCATE all six raw tables (RESTART IDENTITY CASCADE) before loading, "
             "for a clean re-run during development.",
    )
    args = parser.parse_args()
    load_all_data(reset=args.reset)
