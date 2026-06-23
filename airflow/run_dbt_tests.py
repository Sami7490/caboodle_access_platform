"""
run_dbt_tests.py

Runs dbt test (the dbt Core command that executes all data quality
tests we defined in schema.yml) and logs every test result to
raw.dbt_test_results (our data quality monitoring table in Postgres).

This is the data quality monitoring pattern: rather than just running
dbt test manually and reading the terminal output, we capture every
result programmatically so we can:
  - See a trend of pass/fail rates over time in the dashboard
  - Alert on failures automatically
  - Compare today's results to last week's

Run this script manually, or add it as a fourth task in the Airflow
nightly feed DAG so data quality is checked automatically every night
after dbt run completes.
"""

import subprocess
import json
import psycopg2
import os
from datetime import datetime

DBT_PROJECT_DIR = os.path.expanduser(
    "~/Desktop/DE Projects/caboodle_access_platform/dbt/caboodle_access"
)

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "caboodle_access",
    "user": "postgres",
}


def run_dbt_tests():
    """
    Runs dbt test (dbt Core's built-in data quality test runner) and
    captures the output in JSON format for programmatic parsing.
    """
    print("Running dbt test...")

    # --output json tells dbt to write machine-readable results to a
    # file rather than just printing human-readable output to stdout.
    result = subprocess.run(
        ["/Library/Frameworks/Python.framework/Versions/3.13/bin/dbt", "test"],
        cwd=DBT_PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    print(f"dbt test exit code: {result.returncode}")
    return result.stdout, result.returncode


def parse_and_store_results(stdout):
    """
    Parses dbt plain text output and writes one row per test to
    raw.dbt_test_results (our data quality monitoring table).

    dbt plain text output contains lines like:
      PASS not_null_dim_patients_patient_key ............ [PASS in 0.03s]
      FAIL not_null_dim_patients_mrn .................... [FAIL 1]
    We parse these lines to extract test name, model, status, and failures.
    """
    import re
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    rows_inserted = 0
    run_at = datetime.now()

    for line in stdout.strip().split("\n"):
        line = line.strip()

        # Match PASS lines: e.g. "PASS not_null_dim_patients_patient_key"
        pass_match = re.search(r"PASS\s+([\w_]+)", line)
        fail_match = re.search(r"FAIL\s+(\d+)\s+([\w_]+)", line)
        warn_match = re.search(r"WARN\s+(\d+)\s+([\w_]+)", line)

        if pass_match:
            node_name = pass_match.group(1)
            status = "pass"
            failures = 0
            message = None
        elif fail_match:
            failures = int(fail_match.group(1))
            node_name = fail_match.group(2)
            status = "fail"
            message = f"{failures} failing rows"
        elif warn_match:
            failures = int(warn_match.group(1))
            node_name = warn_match.group(2)
            status = "warn"
            message = f"{failures} warned rows"
        else:
            continue

        # dbt test names follow the pattern:
        # test_type__model_name__column_name
        parts = node_name.split("__")
        test_name = parts[0] if parts else node_name
        model_name = parts[1] if len(parts) > 1 else "unknown"
        column_name = parts[2] if len(parts) > 2 else None

        cur.execute(
            """INSERT INTO raw.dbt_test_results
               (run_at, test_name, model_name, column_name, status, failures, message)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (run_at, test_name, model_name, column_name, status, failures, message)
        )
        rows_inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Stored {rows_inserted} test results in raw.dbt_test_results.")
    return rows_inserted


def main():
    stdout, exit_code = run_dbt_tests()
    rows = parse_and_store_results(stdout)

    if rows == 0:
        # Fallback: dbt test output format may vary -- log a summary row
        # so the monitoring table always has a record of each run.
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        status = "pass" if exit_code == 0 else "fail"
        cur.execute(
            """INSERT INTO raw.dbt_test_results
               (test_name, model_name, status, message)
               VALUES (%s, %s, %s, %s)""",
            ("dbt_test_suite", "all_models", status,
             f"dbt test completed with exit code {exit_code}")
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"Stored summary row (exit code {exit_code}).")


if __name__ == "__main__":
    main()
