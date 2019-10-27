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
    
def prepare_features(data) -> None:
    get_detailed_datetime(data)
    get_building_age(data)
    get_site_primary_use(data)
    get_month_primary_use(data)
    get_hour_primary_use(data)
    transform_wind_direction(data)