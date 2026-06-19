# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A synthetic healthcare **Access & Utilization** analytics platform. It generates fake hospital data (patients, encounters, appointments) into local Postgres, then transforms it through dbt into dimensional marts. The domain models real-world clinical metrics: 30-day readmissions, appointment no-shows, length-of-stay, and patient days.

The data is intentionally synthetic but realistic — each synthetic patient carries a *hidden* risk profile that biases the probability of no-shows and readmissions, but that profile is **never written to the database**. Only observable events land in tables, so downstream models/ML must learn risk from observable data alone (see the module docstring in `generator/generate_synthetic_data.py`).

**Project goal:** this is a portfolio project supporting a pivot toward AI Engineer roles — favor production-grade, demonstrable patterns over shortcuts.

## Architecture: the data flow

The pipeline is a linear `raw → staging → marts` flow, where each stage lives in a different tool/directory:

1. **`sql/01_create_schema.sql`** — DDL for the `raw` Postgres schema. Six landing tables modeled on an Epic/Caboodle-style source: `date_dim`, `department_dim`, `provider_dim`, `patient_dim`, `encounter_fact`, `appointment_fact`. These are deliberately "dumb" landing tables — no business logic here.

2. **`generator/generate_synthetic_data.py`** — populates `raw.*` via `psycopg2`. Uses `Faker` + `numpy` weighted sampling, all seeded (`seed=42`) for reproducibility. Key tunables at the top: `NUM_PATIENTS`, `YEARS_OF_HISTORY`. The `load_all_data()` function orchestrates inserts in FK order (date → dept → provider → patient → appointment → encounter).

3. **`dbt/caboodle_access/`** — dbt Core project that builds two downstream schemas:
   - `models/staging/` → materialized as **views** in schema `analytics_staging`. One `stg_*` per raw table; light cleanup + derived columns (e.g. `stg_encounters` adds `length_of_stay_days`, `stg_appointments` adds `lead_time_days`).
   - `models/marts/` → materialized as **tables** in schema `analytics_marts`. Dimensional/fact tables ready for the dashboard, plus derived business logic: `fact_appointments` (adds `is_no_show`), `fct_readmissions` (uses a `LAG()` window over inpatient encounters to compute `is_30_day_readmission`), `dim_patients`, `fact_encounters`.

4. **`agent/`, `airflow/`, `dashboard/`** — empty placeholder directories for planned components (orchestration, BI). Nothing implemented yet.

### Business-logic conventions worth knowing
- **Readmission** = inpatient-to-inpatient only, admission within 30 days of the same patient's *prior inpatient discharge* (standard clinical definition). Lives entirely in `fct_readmissions.sql`, not in raw SQL.
- **No-shows** live in `appointment_fact` and are kept separate from `encounter_fact` because a no-show never produces an encounter.
- Time deltas are computed as `extract(epoch from (a - b)) / 86400.0` to get fractional days — repeated across staging models.
- The `raw_caboodle` dbt source (`models/staging/sources.yml`) maps to the Postgres `raw` schema.
- dbt's custom-schema default **prepends** the profile schema (`analytics`) to each model's `+schema`, so views land in `analytics_staging` and tables in `analytics_marts` (not bare `staging`/`marts`).

## Roadmap / build phases

- **Phase 1 (current):** core platform — Postgres `raw` schema, synthetic generator, dbt staging/marts including readmission logic, plus an Airflow DAG that simulates nightly new appointments/encounters.
- **Phase 2:** predictive layer — scikit-learn models for no-show and readmission, trained only on observable events (never the hidden risk profile).
- **Phase 3:** AI engineering layer — a multi-tool Claude agent replacing single-shot NL-to-SQL, with tools for querying the DB, fetching risk scores, and generating chart specs; a read-only SQL guardrail layer; an eval harness of scored test questions; an LLM-call logging / observability table; and RAG via pgvector over synthetic clinical notes using a local embedding model.
- **Phase 4:** local Streamlit dashboard — charts plus an NL query box backed by the full agent.

## Commands

All dbt commands run from `dbt/caboodle_access/`.

```bash
# 1. Create the raw schema (requires a running local Postgres + caboodle_access db)
# Connect as the postgres role — the default OS-user role does not exist.
psql -U postgres -d caboodle_access -f sql/01_create_schema.sql

# 2. Generate + load synthetic data
python generator/generate_synthetic_data.py

# 3. Build the dbt models
cd dbt/caboodle_access
dbt run                      # build all staging views + marts tables
dbt run --select staging     # build only one layer
dbt run --select fct_readmissions   # build a single model (+ deps with +fct_readmissions)
dbt test                     # run all data tests
dbt test --select fact_appointments # test a single model
dbt build                    # run + test in DAG order
dbt compile                  # render compiled SQL into target/ without executing
```

## Database connection

- dbt profile `caboodle_access` (in `~/.dbt/profiles.yml`): Postgres `localhost:5432`, db `caboodle_access`, user `postgres`, password `epic`, default schema `analytics`, `type: postgres`.
- The generator's `DB_CONFIG` connects to the same db/host/user with the matching password `epic`. The default OS-user role does not exist, so always connect as `postgres` (e.g. `psql -U postgres`).

## Dependencies

There is no requirements file. The generator needs `psycopg2` (or `psycopg2-binary`), `faker`, and `numpy`. dbt needs `dbt-core` + `dbt-postgres`.
