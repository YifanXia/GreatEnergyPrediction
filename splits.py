import pandas as pd
from typing import Tuple, Dict
from params import METER_ID, TARGET_COL
from sklearn.model_selection import train_test_split
import logging

def split_data_by_meter(data: pd.DataFrame) -> Dict:
    meter_data = {}
    for meter_name in METER_ID:
        meter_data[meter_name] = data[data.meter == METER_ID.get(meter_name)]
    return meter_data

def split_train_validation(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, valid_data, _, _ = train_test_split(data, data[TARGET_COL], test_size=0.25, random_state=1)
    
    return (train_data, valid_data)

def time_based_split_train_validation(data: pd.DataFrame, 
                                      test_size_in_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    delta_ts = pd.to_timedelta(f'{test_size_in_days} days')
    split_ts = data.timestamp.max() - delta_ts
    train_data = data[data.timestamp > split_ts]
    valid_data = data[data.timestamp <= split_ts]
    return (train_data, valid_data)

def time_based_half_year_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    delta_ts = pd.to_timedelta('183 days')
    split_ts = data.timestamp.max() - delta_ts
    data_first_half = data[data.timestamp <= split_ts]
    data_second_half = data[data.timestamp > split_ts]
    return (data_first_half, data_second_half)
    
def odd_even_month_half_year_split(data: pd.DataFrame) ->  Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info('Split data into odd / even months.')
    odd_half = data[data.timestamp.dt.month % 2 == 1]
    even_half = data[data.timestamp.dt.month % 2 == 0]
    logging.debug(f'Odd months are: {odd_half.month.unique()}')
    logging.debug(f'Even months are: {even_half.month.unique()}')
    return (odd_half, even_half)