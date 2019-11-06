import pandas as pd
import numpy as np
from typing import Tuple
import logging

def read_building_metadata(path: str) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    metadata.loc[:, 'square_feet'] = np.log(metadata['square_feet'])
    return reduce_mem_usage(metadata)

def read_building_data(path: str, remove_zeros: bool = False) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=['timestamp'])
    if remove_zeros:
        logging.info('Removing zero readings.')
        data = data.query('meter_reading > 0')
    return reduce_mem_usage(data)

def read_weather_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=['timestamp'])
    return reduce_mem_usage(data)

def log_transform_targets(data: pd.DataFrame) -> None:
    data.loc[:, 'log_meter_reading'] = np.log1p(data.loc[:, 'meter_reading'])

def read_data(data_path: str, weather_path: str, meta_data: pd.DataFrame,
              remove_zeros: bool = False, nrows: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    building_data = read_building_data(data_path, remove_zeros)
    weather_data = read_weather_data(weather_path)
    building_meta = building_data.merge(meta_data, on='building_id', how='left')
    #building_weather_meta = building_meta.merge(weather_data, on=['site_id', 'timestamp'], how='left')
    building_meta = reduce_mem_usage(building_meta)
    weather_data = reduce_mem_usage(weather_data)
    return (building_meta, weather_data)

def find_consecutive_repeated_readings(data: pd.DataFrame, window_hours: int = 24) -> None:
    pass

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df.loc[:, col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df.loc[:, col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df.loc[:, col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df.loc[:, col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df.loc[:, col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df.loc[:, col] = df[col].astype(np.float32)
                else:
                    df.loc[:, col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
