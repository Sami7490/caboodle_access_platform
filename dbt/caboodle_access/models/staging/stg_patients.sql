-- Staging model: a cleaned-up, one-to-one version of raw.patient_dim.
-- No heavy logic here -- just clear column selection plus one derived
-- field (current age) that's genuinely useful downstream.

with source as (
    select * from {{ source('raw_caboodle', 'patient_dim') }}
)

select
    patient_key,
    mrn,
    first_name,
    last_name,
    date_of_birth,
    -- age() returns an interval between two dates; date_part('year', ...)
    -- pulls out just the whole-number years component.
    date_part('year', age(current_date, date_of_birth)) as age_years,
    sex,
    race,
    ethnicity,
    zip_code,
    primary_language
from source
