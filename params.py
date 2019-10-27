METER_ID = {
    'Electricity': 0,
    'ChilledWater': 1,
    'Steam': 2,
    'HotWater': 3,
}

FEATURE_LIST = [
    'primary_use',
    'square_feet',
    'year_built',
    'floor_count',
    'air_temperature',
    'cloud_coverage',
    'dew_temperature',
    'precip_depth_1_hr',
    'sea_level_pressure',
    # 'wind_direction',
    'wind_speed',
    'month',
    'day',
    'dayofweek',
    'hour',
    'building_age',
    'site_primary_use',
    'month_primary_use',
    'hour_primary_use',
    'wind_direction_sin',
    ]

CATEGORICAL_LIST = [
    'primary_use',
    'site_primary_use',
    'month_primary_use',
    'hour_primary_use',
]

TARGET_COL = 'log_meter_reading'
