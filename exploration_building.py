import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os

def read_building_metadata(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def plot_building_meter_reading(building_data: pd.DataFrame, site_id: int, metadata: pd.DataFrame) -> None:
    for building_id, primary_use in metadata[metadata.site_id == site_id][['building_id', 'primary_use']].values:
        building_data_id = building_data[building_data.building_id == building_id]
        building_data_id.loc[:, 'log_meter_reading'] = np.log(building_data_id.meter_reading + 1)
        for meter_id in building_data_id.meter.unique():
            building_data_id_meter = building_data_id[building_data_id.meter == meter_id]
            plt.figure(figsize=(14, 5))
            plt.scatter(building_data_id_meter.timestamp, building_data_id_meter.log_meter_reading, marker='.', s=3, c='blue')
            plt.title(f'building: {building_id} {primary_use}, meter_id: {meter_id}')
            if not os.path.exists('figs_site_' + str(site_id)):
                os.mkdir('figs_site_' + str(site_id))
            if not os.path.exists('figs_site_' + str(site_id) + '/meter_' + str(meter_id)):
                os.mkdir('figs_site_' + str(site_id) + '/meter_' + str(meter_id))
            plt.savefig('figs_site_' + str(site_id) + 
                        '/meter_' + str(meter_id) + 
                        '/building_' + str(building_id) + '.png', 
                        format='png')
            plt.close()

def plot_weather_data(weather_data: pd.DataFrame, site_id: int) -> pd.DataFrame:
    pass
        
if __name__ == "__main__":
    
    register_matplotlib_converters()
    data_path = "../data/train.csv"
    meta_data_path = "../data/building_metadata.csv"
    building_data = pd.read_csv(data_path, parse_dates=['timestamp'])
    meta_data = read_building_metadata(meta_data_path)
    for site_id in meta_data.site_id.unique():
        plot_building_meter_reading(building_data, site_id, meta_data)