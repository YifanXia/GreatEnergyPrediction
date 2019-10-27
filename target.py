import pandas as pd
import numpy as np

def get_log_target(data: pd.DataFrame) -> None:
    data.loc[:, 'log_meter_reading'] = np.log(data.meter_reading + 1)