import pandas as pd
import numpy as np
import logging

def get_month(data: pd.DataFrame) -> None:
    data.loc[:, 'month'] = data.timestamp.dt.month
    
def get_day(data: pd.DataFrame) -> None:
    data.loc[:, 'day'] = data.timestamp.dt.day

def get_dayofyear(data: pd.DataFrame) -> None:
    data.loc[:, 'dayofyear'] = data.timestamp.dt.dayofyear
    
def get_dayofweek(data: pd.DataFrame) -> None:
    data.loc[:, 'dayofweek'] = data.timestamp.dt.dayofweek
    
def get_hour(data: pd.DataFrame) -> None:
    data.loc[:, 'hour'] = data.timestamp.dt.hour
    
def get_detailed_datetime(data: pd.DataFrame) -> None:
    get_month(data)
    get_dayofyear(data)
    get_day(data)
    get_dayofweek(data)
    get_hour(data)

def get_building_age(data: pd.DataFrame) -> None:
    data.loc[:, 'building_age'] = data.timestamp.dt.year - data.year_built
    
def get_site_primary_use(data: pd.DataFrame) -> None:
    data.loc[:, 'site_primary_use'] = data.site_id.map(str) + '_' + data.primary_use

def get_hour_primary_use(data: pd.DataFrame) -> None:
    data.loc[:, 'hour_primary_use'] = data.hour.map(str) + '_' + data.primary_use
    
def get_month_primary_use(data: pd.DataFrame) -> None:
    data.loc[:, 'month_primary_use'] = data.month.map(str) + '_' + data.primary_use

def transform_wind_direction(data: pd.DataFrame) -> None:
    data.loc[:, 'wind_direction_sin'] = pd.Series(np.sin(np.pi * data.wind_direction.values / 360))

def add_wind_chill(data: pd.DataFrame) -> None:
    def compute_wind_chill(row: pd.Series) -> float:
        v = row['wind_speed'] * 3.6
        T_a = row['air_temperature']
        if T_a <= 10 and v >= 4.8:
            return 13.12 + 0.6215 * T_a - 11.37 * v ** 0.16 + 0.3965 * T_a * v ** 0.16
        return float('nan')
    data.loc[:, 'wind_chill'] = data.apply(compute_wind_chill, axis=1)

def add_relative_humidity(data: pd.DataFrame) -> None:
    def compute_relative_humidity(row: pd.Series) -> float:
        # Magnus formula
        T_a = row['air_temperature']
        T_dp = row['dew_temperature']
        b = 18.678
        c = 257.14 # Celsius
        return 100 * np.exp(T_dp * b / (T_dp + c) - b * T_a / (c + T_a))
    data.loc[:, 'relative_humidity'] = data.apply(compute_relative_humidity, axis=1)
    
def add_heat_index(data: pd.DataFrame) -> None:
    def compute_heat_index(row: pd.Series) -> float:
        T = row['air_temperature']
        R = row['relative_humidity']
        if R >= 40.0 and T >= 27.0:
            c = [0] * 9
            c[0] = -8.78469475556
            c[1] = 1.61139411
            c[2] = 2.33854883889
            c[3] = -0.14611605
            c[4] = -0.012308094
            c[5] = -0.0164248277778
            c[6] = 0.002211732
            c[7] = 0.00072546
            c[8] = -0.000003582
        
            vals = [1] * 9
            vals[1] = T
            vals[2] = R
            vals[3] = T * R
            vals[4] = T **2
            vals[5] = R **2
            vals[6] = T **2 * R
            vals[7] = T * R **2
            vals[8] = (T * R) **2
        
            return sum(a * b for a, b in zip(c, vals))
        return float('nan')
    data.loc[:, 'heat_index'] = data.apply(compute_heat_index, axis=1)

def add_feels_like(data: pd.DataFrame, remove_others: bool = False) -> None:
    add_relative_humidity(data)
    add_heat_index(data)
    add_wind_chill(data)
    data.loc[:, 'feels_like'] = data['heat_index'].combine_first(data['wind_chill']).combine_first(data['air_temperature'])
    if remove_others:
        data.drop(['heat_index', 'wind_chill'], axis=1)

def add_clear_sky_index(data: pd.DataFrame) -> None:
    data.loc[:, 'clear_sky_index'] = 1 - 0.75 * (data['cloud_coverage'] / 8) ** 3

def fill_na_in_weather_data(weather_data: pd.DataFrame) -> None:
    weather_data = weather_data.fillna(method='backfill').fillna(method='ffill')

def add_lag_features(weather_data: pd.DataFrame, window_size: int = 24) -> None:
    group_df = weather_data.groupby('site_id')
    cols_mean = [
        'air_temperature', 
        'dew_temperature', 
        'precip_depth_1_hr', 
        'sea_level_pressure', 
        'wind_speed', 
        'relative_humidity', 
        'feels_like',
        'clear_sky_index',
        ]
    cols_median = ['cloud_coverage']
    rolled_mean = group_df[cols_mean].rolling(window=window_size, min_periods=0, win_type='triang')
    rolled_median = group_df[cols_median].rolling(window=window_size, min_periods=0, win_type=None)
    lag_mean = rolled_mean.mean().reset_index().astype(np.float16)
    lag_median = rolled_median.median().reset_index().astype(np.float16)

    for col in cols_mean:
        weather_data[f'{col}_mean_lag{window_size}'] = lag_mean[col]   
    for col in cols_median:
        weather_data[f'{col}_median_lag{window_size}'] = lag_median[col]
    
    
def prepare_features(data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    get_detailed_datetime(data)
    get_building_age(data)
    get_site_primary_use(data)
    get_month_primary_use(data)
    get_hour_primary_use(data)
    ##fill_na_in_weather_data(weather_data)
    add_feels_like(weather_data)
    add_clear_sky_index(weather_data)
    add_lag_features(weather_data, window_size=12)
    add_lag_features(weather_data, window_size=24)
    add_lag_features(weather_data, window_size=36)
    transform_wind_direction(weather_data)
    return data.merge(weather_data, on=['site_id', 'timestamp'], how='left')