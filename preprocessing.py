import pandas as pd
import numpy as np

def read_building_metadata(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def log_transform_targets(data: pd.DataFrame) -> None:
    data.loc[:, 'log_meter_reading'] = np.log(data.loc[:, 'meter_reading'])

def read_data(data_path: str, weather_path: str, meta_data: pd.DataFrame) -> pd.DataFrame:
    building_data = pd.read_csv(data_path, parse_dates='timestamp')
    weather_data = pd.read_csv(weather_path, parse_dates='timestamp')
    building_meta = building_data.merge(meta_data, on='building_id', how='left')
    building_weather_meta = building_meta.merge(weather_data, on=['site_id', 'timestamp'], how='left')
    return building_weather_meta