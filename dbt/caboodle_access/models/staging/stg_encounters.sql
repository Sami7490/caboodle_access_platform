-- Staging model: cleaned-up version of raw.encounter_fact.
-- Adds a length_of_stay_days field, calculated from the admission and
-- discharge timestamps -- used directly for the "Patient Days" metric.

with source as (
    select * from {{ source('raw_caboodle', 'encounter_fact') }}
)

select
    encounter_key,
    encounter_id,
    patient_key,
    department_key,
    provider_key,
    encounter_type,
    admission_datetime,
    discharge_datetime,
    discharge_disposition,
    -- extract(epoch from ...) gives the interval in total seconds;
    -- dividing by 86400 (seconds per day) converts that to days,
    -- including fractional days for short ED visits.
    extract(epoch from (discharge_datetime - admission_datetime)) / 86400.0 as length_of_stay_days
from source
