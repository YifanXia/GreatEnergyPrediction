METER_ID = {
    'Electricity': 0,
    'ChilledWater': 1,
    'Steam': 2,
    'HotWater': 3,
}

FEATURE_LIST = [
    'building_id',
    'site_id',
    'primary_use',
    'square_feet',
    'year_built',
    'floor_count',
    'month',
    'day',
    'dayofweek',
    'hour',
    'building_age',
    'site_primary_use',
    'month_primary_use',
    'hour_primary_use',
    'air_temperature',
    'cloud_coverage',
    'dew_temperature',
    'precip_depth_1_hr',
    'sea_level_pressure',
    #'wind_direction',
    'wind_speed',
    'air_temperature_mean_lag24',
    'dew_temperature_mean_lag24',
    'precip_depth_1_hr_mean_lag24',
    'sea_level_pressure_mean_lag24',
    'wind_speed_mean_lag24',
    'cloud_coverage_median_lag24',
    'wind_direction_sin',
    ]

CATEGORICAL_LIST = [
    'building_id',
    'site_id',
    'primary_use',
    'site_primary_use',
    'month_primary_use',
    'hour_primary_use',
]

TARGET_COL = 'log_meter_reading'
