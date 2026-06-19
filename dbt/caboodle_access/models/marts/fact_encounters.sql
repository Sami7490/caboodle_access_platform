-- Marts model: encounter fact table, ready for direct dashboard use.
-- Joins in the 30-day readmission flag for inpatient encounters --
-- non-inpatient encounters simply get FALSE (they're not eligible).

select
    e.encounter_key,
    e.encounter_id,
    e.patient_key,
    e.department_key,
    e.provider_key,
    e.encounter_type,
    e.admission_datetime,
    e.discharge_datetime,
    e.discharge_disposition,
    e.length_of_stay_days,
    coalesce(r.is_30_day_readmission, false) as is_30_day_readmission
from {{ ref('stg_encounters') }} e
left join {{ ref('fct_readmissions') }} r
    on e.encounter_key = r.encounter_key
