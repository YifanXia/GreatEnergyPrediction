import logging
import numpy as np
import pandas as pd
import preprocessing as pp
import features
import target
import splits
from LgbmTrainer import LgbmModel

MODEL_CODE = '02'
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    TRAIN_DATA_PATH = "../data/train.csv"
    TRAIN_WEATHER_PATH = "../data/weather_train.csv"
    TEST_DATA_PATH = "../data/test.csv"
    TEST_WEATHER_PATH = "../data/weather_test.csv"
    META_DATA_PATH = "../data/building_metadata.csv"
    
    meta_data = pp.read_building_metadata(META_DATA_PATH)
    train_building_data, train_weather_data = pp.read_data(TRAIN_DATA_PATH, TRAIN_WEATHER_PATH, meta_data, nrows=None)
    
    logging.info('Preparing training features and target')
    training_data = features.prepare_features(train_building_data, train_weather_data)
    del train_building_data, train_weather_data
    target.get_log_target(training_data)
    logging.info('Splitting data by meter type...')
    train_sets = splits.split_data_by_meter(training_data)
    del training_data
    logging.info('Training set ready.')
    logging.info(f'meters are {list(train_sets.keys())} in training')
    ## Train model
    meter_models = {}
    logging.info('Training Models')
    for meter_type in train_sets:
        model = LgbmModel()
        model.train(train_sets[meter_type])
        logging.info(f'Training {meter_type} meter model.')
        meter_models[meter_type] = model
        model.save_model(f'{meter_type}_model_{MODEL_CODE}.pkl')
    
    del train_sets
    logging.info('Training finished.')
    logging.info('')
    
    
    
    logging.info('Reading test data')
    test_building_data, test_weather_data = pp.read_data(TEST_DATA_PATH, TEST_WEATHER_PATH, meta_data, nrows=None)
    test_data = features.prepare_features(test_building_data, test_weather_data)
    del test_building_data, test_weather_data
    del meta_data
    
    test_sets = splits.split_data_by_meter(test_data)
    del test_data
    logging.info('Test set ready.')
    logging.info(f'meters are {list(test_sets.keys())} in test')
    ## Predict
    meter_predictions = {}
    for meter_type in meter_models:
        logging.info(f'Making predictions for {meter_type} meter...')
        try:
            predictions = meter_models[meter_type].predict(test_sets[meter_type])
            test_sets[meter_type].loc[:, 'meter_reading'] = np.maximum(np.exp(predictions) - 1, 0)
        except ValueError:
            test_sets[meter_type] = None
    
    logging.info('Writting predictions')
    pd.concat([data[['row_id', 'meter_reading']] for data in test_sets.values() if data is not None])\
        .sort_values('row_id').to_csv(f'results_{MODEL_CODE}.csv', index=False)
    