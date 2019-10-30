import pandas as pd
import numpy as np

def get_month(data: pd.DataFrame) -> None:
    data.loc[:, 'month'] = data.timestamp.dt.month
    
def get_day(data: pd.DataFrame) -> None:
    data.loc[:, 'day'] = data.timestamp.dt.day
    
def get_dayofweek(data: pd.DataFrame) -> None:
    data.loc[:, 'dayofweek'] = data.timestamp.dt.dayofweek
    
def get_hour(data: pd.DataFrame) -> None:
    data.loc[:, 'hour'] = data.timestamp.dt.hour
    
def get_detailed_datetime(data: pd.DataFrame) -> None:
    get_month(data)
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

def fill_na_in_weather_data(weather_data: pd.DataFrame) -> None:
    weather_data = weather_data.fillna(method='backfill').fillna(method='ffill')

def add_yesterday_lag_features(weather_data: pd.DataFrame, window_size: int = 24) -> None:
    group_df = weather_data.groupby('site_id')
    cols_mean = ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed']
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
    add_yesterday_lag_features(weather_data)
    transform_wind_direction(weather_data)
    return data.merge(weather_data, on=['site_id', 'timestamp'], how='left')