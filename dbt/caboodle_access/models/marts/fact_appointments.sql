-- Marts model: appointment fact table, ready for direct dashboard use.
-- Adds an is_no_show boolean flag for easy aggregation.

select
    appointment_key,
    appointment_id,
    patient_key,
    department_key,
    provider_key,
    scheduled_datetime,
    booked_datetime,
    appointment_status,
    appointment_type,
    lead_time_days,
    case when appointment_status = 'No Show' then true else false end as is_no_show
from {{ ref('stg_appointments') }}
