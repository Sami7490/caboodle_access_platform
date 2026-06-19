-- ============================================================================
-- 01_create_schema.sql
--
-- Creates the RAW schema for our synthetic Access & Utilization database.
-- These are "raw" landing tables -- close to how data would arrive from a
-- source system. Clean modeling (types, business logic, the readmission
-- flag) happens downstream in dbt Core, not here.
-- ============================================================================

-- A dedicated schema (a namespace inside Postgres) called "raw" keeps these
-- landing tables clearly separate from the "staging"/"marts" schemas dbt
-- will build later.
CREATE SCHEMA IF NOT EXISTS raw;

-- ----------------------------------------------------------------------------
-- date_dim: standard date dimension. Every fact table joins to this so we
-- can roll up by month/quarter/day-of-week without doing date math at query
-- time.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.date_dim (
    date_key        INT PRIMARY KEY,        -- surrogate key, format YYYYMMDD
    calendar_date   DATE NOT NULL,
    day_of_week     TEXT NOT NULL,           -- 'Monday', 'Tuesday', etc.
    day_of_week_num INT NOT NULL,            -- 1 = Monday ... 7 = Sunday
    is_weekend      BOOLEAN NOT NULL,
    month_num       INT NOT NULL,
    month_name      TEXT NOT NULL,
    quarter_num     INT NOT NULL,
    year_num        INT NOT NULL
);

-- ----------------------------------------------------------------------------
-- department_dim: clinics/units where appointments and encounters happen.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.department_dim (
    department_key   SERIAL PRIMARY KEY,     -- auto-incrementing surrogate key
    department_name  TEXT NOT NULL,
    department_type  TEXT NOT NULL,          -- 'Outpatient Clinic', 'Inpatient Unit', 'Emergency Department'
    specialty        TEXT
);

-- ----------------------------------------------------------------------------
-- provider_dim: clinicians who see patients / are scheduled for visits.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.provider_dim (
    provider_key     SERIAL PRIMARY KEY,
    provider_name    TEXT NOT NULL,
    provider_type    TEXT NOT NULL,          -- 'MD', 'NP', 'PA'
    specialty        TEXT
);

-- ----------------------------------------------------------------------------
-- patient_dim: master patient demographic table.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.patient_dim (
    patient_key      SERIAL PRIMARY KEY,
    mrn              TEXT NOT NULL UNIQUE,   -- medical record number (business key)
    first_name       TEXT NOT NULL,
    last_name        TEXT NOT NULL,
    date_of_birth    DATE NOT NULL,
    sex              TEXT NOT NULL,
    race             TEXT,
    ethnicity        TEXT,
    zip_code         TEXT,
    primary_language TEXT
);

-- ----------------------------------------------------------------------------
-- encounter_fact: inpatient admissions, ED visits, outpatient visits.
-- The 30-day readmission flag will be derived from this table in dbt.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.encounter_fact (
    encounter_key          SERIAL PRIMARY KEY,
    encounter_id           TEXT NOT NULL UNIQUE,   -- business key, like Epic's CSN
    patient_key            INT NOT NULL REFERENCES raw.patient_dim(patient_key),
    department_key         INT NOT NULL REFERENCES raw.department_dim(department_key),
    provider_key           INT REFERENCES raw.provider_dim(provider_key),
    encounter_type         TEXT NOT NULL,       -- 'Inpatient', 'Emergency', 'Outpatient'
    admission_datetime     TIMESTAMP NOT NULL,
    discharge_datetime     TIMESTAMP,
    discharge_disposition  TEXT                  -- 'Home', 'SNF', 'Expired', 'AMA', etc.
);

-- ----------------------------------------------------------------------------
-- appointment_fact: scheduled outpatient appointments, including no-shows.
-- Kept separate from encounter_fact since a no-show never produces an
-- encounter.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.appointment_fact (
    appointment_key      SERIAL PRIMARY KEY,
    appointment_id        TEXT NOT NULL UNIQUE,
    patient_key            INT NOT NULL REFERENCES raw.patient_dim(patient_key),
    department_key         INT NOT NULL REFERENCES raw.department_dim(department_key),
    provider_key            INT REFERENCES raw.provider_dim(provider_key),
    scheduled_datetime       TIMESTAMP NOT NULL,   -- when the visit was scheduled FOR
    booked_datetime           TIMESTAMP NOT NULL,   -- when the appointment was MADE (for lead-time calc)
    appointment_status        TEXT NOT NULL,        -- 'Completed', 'No Show', 'Cancelled', 'Rescheduled'
    appointment_type           TEXT NOT NULL         -- 'New Patient', 'Follow-up', etc.
);

-- Indexes to speed up the joins/filters we'll run most often.
CREATE INDEX IF NOT EXISTS idx_encounter_patient   ON raw.encounter_fact(patient_key);
CREATE INDEX IF NOT EXISTS idx_encounter_admit     ON raw.encounter_fact(admission_datetime);
CREATE INDEX IF NOT EXISTS idx_appointment_patient ON raw.appointment_fact(patient_key);
CREATE INDEX IF NOT EXISTS idx_appointment_sched   ON raw.appointment_fact(scheduled_datetime);



