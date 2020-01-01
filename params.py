METER_ID = {
    'Electricity': 0,
    'ChilledWater': 1,
    'Steam': 2,
    'HotWater': 3,
}

FEATURE_LIST = [
    'building_id',
    #'site_id',
    'meter',
    #'primary_use',
    #'square_feet',
    #'year_built',
    #'floor_count',
    #'month',
    #'day',
    #'dayofyear',
    'dayofweek',
    #'is_weekend',
    'hour',
    #'building_age',
    'site_primary_use',
    #'month_primary_use',
    #'hour_primary_use',
    'air_temperature',
    #'cloud_coverage',
    #'dew_temperature',
    #'precip_depth_1_hr',
    'sea_level_pressure',
    #'wind_direction',
    'relative_humidity',
    'feels_like',
    #'clear_sky_index',
    #'wind_speed',
    #'air_temperature_mean_lag12',
    #'dew_temperature_mean_lag12',
    #'precip_depth_1_hr_mean_lag12',
    #'sea_level_pressure_mean_lag12',
    #'wind_speed_mean_lag12',
    #'relative_humidity_mean_lag12',
    #'feels_like_mean_lag12',
    #'clear_sky_index_mean_lag12',
    #'cloud_coverage_median_lag12',
    
    #'air_temperature_mean_lag24',
    #'dew_temperature_mean_lag24',
    #'precip_depth_1_hr_mean_lag24',
    #'sea_level_pressure_mean_lag24',
    #'wind_speed_mean_lag24',
    #'relative_humidity_mean_lag24',
    #'feels_like_mean_lag24',
    #'clear_sky_index_mean_lag24',
    #'cloud_coverage_median_lag24',
    
    #'air_temperature_mean_lag36',
    #'dew_temperature_mean_lag36',
    #'precip_depth_1_hr_mean_lag36',
    #'sea_level_pressure_mean_lag36',
    #'wind_speed_mean_lag36',
    #'relative_humidity_mean_lag36',
    #'feels_like_mean_lag36',
    #'clear_sky_index_mean_lag36',
    #'cloud_coverage_median_lag36',
    #'wind_direction_sin',
    ]

CATEGORICAL_LIST = [
    'building_id',
    #'site_id',
    'meter',
    #'primary_use',
    'hour',
    'dayofweek',
    #'is_weekend',
    'site_primary_use',
    #'month_primary_use',
    #'hour_primary_use',
]

TARGET_COL = 'log_meter_reading'
