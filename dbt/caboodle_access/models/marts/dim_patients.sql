-- Marts model: the patient dimension, ready for direct dashboard use.

select
    patient_key,
    mrn,
    first_name,
    last_name,
    date_of_birth,
    age_years,
    sex,
    race,
    ethnicity,
    zip_code,
    primary_language
from {{ ref('stg_patients') }}
