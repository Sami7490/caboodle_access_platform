-- Staging model: cleaned-up version of raw.department_dim.

with source as (
    select * from {{ source('raw_caboodle', 'department_dim') }}
)

select
    department_key,
    department_name,
    department_type,
    specialty
from source
