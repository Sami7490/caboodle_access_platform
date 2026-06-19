-- For each INPATIENT encounter, computes whether it represents a 30-day
-- readmission -- whether this admission happened within 30 days of the
-- same patient's previous inpatient discharge. Inpatient-to-inpatient
-- only, matching the standard clinical definition.

with inpatient_encounters as (
    select *
    from {{ ref('stg_encounters') }}
    where encounter_type = 'Inpatient'
),

with_prior_discharge as (
    select
        *,
        -- LAG() looks at a prior row's value within the same partition.
        -- Partitioned by patient, ordered by admission date, this pulls
        -- each patient's PREVIOUS inpatient discharge date onto the
        -- current row.
        lag(discharge_datetime) over (
            partition by patient_key
            order by admission_datetime
        ) as prior_discharge_datetime
    from inpatient_encounters
)

select
    encounter_key,
    encounter_id,
    patient_key,
    department_key,
    provider_key,
    admission_datetime,
    discharge_datetime,
    discharge_disposition,
    length_of_stay_days,
    prior_discharge_datetime,
    -- Gap in days between this admission and the patient's prior
    -- discharge. NULL if this is the patient's first-ever inpatient stay.
    extract(epoch from (admission_datetime - prior_discharge_datetime)) / 86400.0 as days_since_prior_discharge,
    -- TRUE only if there WAS a prior discharge, and it happened 30 days
    -- or less before this admission.
    case
        when prior_discharge_datetime is not null
             and admission_datetime - prior_discharge_datetime <= interval '30 days'
        then true
        else false
    end as is_30_day_readmission
from with_prior_discharge
