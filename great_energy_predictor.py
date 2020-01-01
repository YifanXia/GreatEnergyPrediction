import logging
import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
import preprocessing as pp
import features
import target
import splits
from LgbmTrainer import LgbmModel

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

MODEL_CODE = '23'
if not os.path.exists(f'model_{MODEL_CODE}'):
    os.mkdir(f'model_{MODEL_CODE}')
logging.basicConfig(level=logging.DEBUG,
                    #filename=f'model_{MODEL_CODE}/log_file_{MODEL_CODE}.log',
                    format='%(asctime)s %(levelname)s - %(message)s',
                    datefmt='%m-%d %H:%M')
#sys.stdout = open(f'model_{MODEL_CODE}/log_file_{MODEL_CODE}.log', 'w')

TRAIN_DATA_PATH = "../data/train.csv"
TRAIN_WEATHER_PATH = "../data/weather_train.csv"
TEST_DATA_PATH = "../data/test.csv"
TEST_WEATHER_PATH = "../data/weather_test.csv"
META_DATA_PATH = "../data/building_metadata.csv"

def preprocess_train(meta_data: pd.DataFrame, remove_zeros: bool) -> pd.DataFrame:
    #meta_data = pp.read_building_metadata(META_DATA_PATH)
    logging.info('Reading training data')
    train_building_data, train_weather_data = pp.read_data(TRAIN_DATA_PATH, 
                                                           TRAIN_WEATHER_PATH, 
                                                           meta_data, 
                                                           remove_zeros=remove_zeros, 
                                                           nrows=None)
    logging.info('Preparing training features and target')
    training_data = features.prepare_features(train_building_data, train_weather_data)
    target.get_log_target(training_data)
    #logging.info('Splitting data by meter type...')
    #train_sets = splits.split_data_by_meter(training_data)
    logging.info('Training set ready.')
    return training_data

def preprocess_test(meta_data: pd.DataFrame) -> pd.DataFrame:
    logging.info('Reading test data')
    test_building_data, test_weather_data = pp.read_data(TEST_DATA_PATH, TEST_WEATHER_PATH, meta_data, nrows=None)
    test_data = features.prepare_features(test_building_data, test_weather_data)
    #test_sets = splits.split_data_by_meter(test_data)
    logging.info('Test set ready.')
    return test_data
    
def run_training_pipeline(meta_data: pd.DataFrame) -> LgbmModel:
    train_set = preprocess_train(meta_data, True)
    logging.info('Training Models')
    if not os.path.exists(f'model_{MODEL_CODE}'):
        os.mkdir(f'model_{MODEL_CODE}')
    model = LgbmModel()
    logging.info(f'Training energy prediction model.')
    model.train(train_set)
    model.save_model(f'model_{MODEL_CODE}/model_{MODEL_CODE}.pkl')
    model.plot_feature_importances(f'model_{MODEL_CODE}/feature_{MODEL_CODE}.png')
    logging.info('Training finished.')
    return model

def run_prediction_pipeline(meta_data: pd.DataFrame, model: LgbmModel) -> None:
    test_set = preprocess_test(meta_data)
    logging.info(f'Making predictions...')
    #predictions = model.predict(test_set)
    test_set.loc[:, 'meter_reading'] = model.predict(test_set)#np.clip(np.expm1(predictions), 0, None)
    logging.info('Writting predictions...')
    test_set[['row_id', 'meter_reading']].sort_values('row_id').to_csv(f'results_{MODEL_CODE}.csv', index=False)
    
if __name__ == "__main__":
    meta_data = pp.read_building_metadata(META_DATA_PATH)
    model = run_training_pipeline(meta_data)
    #model = LgbmModel()
    #model.read_model('model_15/model_15.pkl')
    run_prediction_pipeline(meta_data, model)
    #pd.read_csv('results_15.csv')[['row_id', 'meter_reading']].sort_values('row_id').to_csv(f'results_15.csv', index=False)
    
    

    
    