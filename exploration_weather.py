import pandas as pd
from pandas.plotting import register_matplotlib_converters
import os
import matplotlib.pyplot as plt
from preprocessing import read_weather_data
from features import add_feels_like

def plot_one_site(weather_data: pd.DataFrame, site_id: int, weather: str) -> None:
    data = weather_data[weather_data.site_id == site_id]
    plt.figure(figsize=(14, 5))
    plt.scatter(data['timestamp'], data[weather], marker='.', s=3, c='blue')
    plt.title(f'site: {site_id}, weather: {weather}')
    if not os.path.exists(f'figs_site_{site_id}'):
        os.mkdir(f'figs_site_{site_id}')
    if not os.path.exists(f'figs_site_{site_id}/weather'):
        os.mkdir(f'figs_site_{site_id}/weather')
    plt.savefig(f'figs_site_{site_id}/weather/{weather}.png', format='png')
    plt.close()
    
def plot_weather_by_site(weather_data: pd.DataFrame) -> None:
    for site_id in weather_data['site_id'].unique():
        for weather in weather_data.columns.difference(['site_id', 'timestamp']):
            plot_one_site(weather_data, site_id, weather)

if __name__ == "__main__":
    register_matplotlib_converters()
    weather_data = read_weather_data('../data/weather_train.csv')
    add_feels_like(weather_data)
    plot_weather_by_site(weather_data)