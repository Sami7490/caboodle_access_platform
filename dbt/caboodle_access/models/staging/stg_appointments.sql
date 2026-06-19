-- Staging model: cleaned-up version of raw.appointment_fact.
-- Adds a lead_time_days field -- how far in advance the appointment was
-- booked relative to the actual visit. This is a key predictive feature
-- for no-show risk.

with source as (
    select * from {{ source('raw_caboodle', 'appointment_fact') }}
)

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
    extract(epoch from (scheduled_datetime - booked_datetime)) / 86400.0 as lead_time_days
from source
