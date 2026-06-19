-- Staging model: cleaned-up version of raw.date_dim.

with source as (
    select * from {{ source('raw_caboodle', 'date_dim') }}
)

select
    date_key,
    calendar_date,
    day_of_week,
    day_of_week_num,
    is_weekend,
    month_num,
    month_name,
    quarter_num,
    year_num
from source
