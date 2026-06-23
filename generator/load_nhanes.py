"""
load_nhanes.py

Loads NHANES XPT (SAS transport) files into Postgres using pyreadstat
(a Python library that reads SAS/SPSS/Stata file formats) and psycopg2
(our Python-to-Postgres connector).

Each NHANES file becomes its own table in the raw schema, named after
the file (e.g. DEMO_E.xpt -> raw.demo_e). SEQN is the participant ID
that links all tables together -- the equivalent of a patient_key.
"""

import os
import pyreadstat
import psycopg2
import pandas as pd
from psycopg2 import sql

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "nhanes_platform",
    "user": "postgres",
}

# XPT files to load -- all from the 2007-2008 (cycle E) survey
NHANES_FILES = {
    "demo_e":  "/Users/sami/Downloads/DEMO_E.xpt",   # demographics
    "hiq_e":   "/Users/sami/Downloads/HIQ_E.xpt",    # health insurance
    "huq_e":   "/Users/sami/Downloads/HUQ_E.xpt",    # hospital utilization
    "mcq_e":   "/Users/sami/Downloads/MCQ_E.xpt",    # medical conditions
    "bpq_e":   "/Users/sami/Downloads/BPQ_E.xpt",    # blood pressure
    "diq_e":   "/Users/sami/Downloads/DIQ_E.xpt",    # diabetes
    "smq_e":   "/Users/sami/Downloads/SMQ_E.xpt",    # smoking
    "paq_e":   "/Users/sami/Downloads/PAQ_E.xpt",    # physical activity
    "dpq_e":   "/Users/sami/Downloads/DPQ_E.xpt",    # depression screener
}


def load_xpt_to_postgres(table_name, file_path, cur, conn):
    """
    Reads one XPT file using pyreadstat and loads it into Postgres as
    a table in the raw schema. Columns are lowercased for SQL friendliness.
    All values stored as TEXT to avoid type inference issues -- dbt
    will handle proper typing in the transformation layer.
    """
    print(f"  Loading {file_path}...")
    df, meta = pyreadstat.read_xport(file_path)

    # Lowercase all column names for SQL friendliness
    df.columns = [c.lower() for c in df.columns]

    # Drop the table if it already exists and recreate it
    cur.execute(f"DROP TABLE IF EXISTS raw.{table_name}")

    # Build CREATE TABLE with all columns as TEXT
    col_defs = ", ".join([f"{col} TEXT" for col in df.columns])
    cur.execute(f"CREATE TABLE raw.{table_name} ({col_defs})")

    # Insert rows in batches of 1000 for efficiency
    rows = df.astype(str).replace("nan", None).values.tolist()
    batch_size = 1000
    placeholders = ", ".join(["%s"] * len(df.columns))

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cur.executemany(
            f"INSERT INTO raw.{table_name} VALUES ({placeholders})",
            batch
        )

    conn.commit()
    print(f"    -> {len(df)} rows, {len(df.columns)} columns loaded into raw.{table_name}")


def main():
    print("Connecting to nhanes_platform database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Create raw schema
    cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.commit()
    print("Created raw schema.")

    # Load each NHANES file
    for table_name, file_path in NHANES_FILES.items():
        if os.path.exists(file_path):
            load_xpt_to_postgres(table_name, file_path, cur, conn)
        else:
            print(f"  WARNING: {file_path} not found, skipping.")

    cur.close()
    conn.close()
    print("\nAll NHANES files loaded successfully.")


if __name__ == "__main__":
    main()
