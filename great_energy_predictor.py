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

MODEL_CODE = '05'
if not os.path.exists(f'model_{MODEL_CODE}'):
    os.mkdir(f'model_{MODEL_CODE}')
logging.basicConfig(level=logging.INFO,
                    #filename=f'model_{MODEL_CODE}/log_file_{MODEL_CODE}.log',
                    format='%(asctime)s %(levelname)s - %(message)s',
                    datefmt='%m-%d %H:%M')
#sys.stdout = open(f'model_{MODEL_CODE}/log_file_{MODEL_CODE}.log', 'w')

TRAIN_DATA_PATH = "../data/train.csv"
TRAIN_WEATHER_PATH = "../data/weather_train.csv"
TEST_DATA_PATH = "../data/test.csv"
TEST_WEATHER_PATH = "../data/weather_test.csv"
META_DATA_PATH = "../data/building_metadata.csv"

def preprocess_train(meta_data: pd.DataFrame, remove_zeros: bool) -> Dict:
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
    logging.info('Splitting data by meter type...')
    train_sets = splits.split_data_by_meter(training_data)
    logging.info('Training set ready.')
    return train_sets

def preprocess_test(meta_data: pd.DataFrame) -> Dict:
    logging.info('Reading test data')
    test_building_data, test_weather_data = pp.read_data(TEST_DATA_PATH, TEST_WEATHER_PATH, meta_data, nrows=None)
    test_data = features.prepare_features(test_building_data, test_weather_data)
    test_sets = splits.split_data_by_meter(test_data)
    logging.info('Test set ready.')
    return test_sets
    
def run_training_pipeline(meta_data: pd.DataFrame) -> Dict[str, LgbmModel]:
    train_sets = preprocess_train(meta_data, True)
    meter_models = {}
    logging.info('Training Models')
    if not os.path.exists(f'model_{MODEL_CODE}'):
        os.mkdir(f'model_{MODEL_CODE}')
    for meter_type in train_sets:
        model = LgbmModel()
        logging.info(f'Training {meter_type} meter model.')
        model.train(train_sets[meter_type], use_time_based_split=True)
        meter_models[meter_type] = model
        model.save_model(f'model_{MODEL_CODE}/{meter_type}_model_{MODEL_CODE}.pkl')
    logging.info('Training finished.')
    return meter_models

def run_prediction_pipeline(meta_data: pd.DataFrame, meter_models: Dict[str, pd.DataFrame]) -> None:
    test_sets = preprocess_test(meta_data)
    meter_predictions = {}
    for meter_type in meter_models:
        logging.info(f'Making predictions for {meter_type} meter...')
        try:
            predictions = meter_models[meter_type].predict(test_sets[meter_type])
            test_sets[meter_type].loc[:, 'meter_reading'] = np.maximum(np.exp(predictions) - 1, 0)
        except ValueError:
            test_sets[meter_type] = None
    
    logging.info('Writting predictions...')
    pd.concat([data[['row_id', 'meter_reading']] for data in test_sets.values() if data is not None])\
        .sort_values('row_id').to_csv(f'results_{MODEL_CODE}.csv', index=False)
    
if __name__ == "__main__":
    meta_data = pp.read_building_metadata(META_DATA_PATH)
    meter_models = run_training_pipeline(meta_data)
    run_prediction_pipeline(meta_data, meter_models)
    
    

    
    