import pandas as pd
from typing import Tuple, Dict
from params import METER_ID, TARGET_COL
from sklearn.model_selection import train_test_split

def split_data_by_meter(data: pd.DataFrame) -> Dict:
    meter_data = {}
    for meter_name in METER_ID:
        meter_data[meter_name] = data[data.meter == METER_ID.get(meter_name)]
    return meter_data

def split_train_validation(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, valid_data, _, _ = train_test_split(data, data[TARGET_COL], test_size=0.25, random_state=1)
    
    return (train_data, valid_data)
