-- Staging model: cleaned-up version of raw.provider_dim.

with source as (
    select * from {{ source('raw_caboodle', 'provider_dim') }}
)

select
    provider_key,
    provider_name,
    provider_type,
    specialty
from source
