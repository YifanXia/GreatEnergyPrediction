import preprocessing as pp
import features
import target
import splits

if __name__ == "__main__":
    TRAIN_DATA_PATH = "../data/train.csv"
    TRAIN_WEATHER_PATH = "../data/weather_train.csv"
    TEST_DATA_PATH = "../data/test.csv"
    TEST_WEATHER_PATH = "../data/weather_test.csv"
    META_DATA_PATH = "../data/building_metadata.csv"
    
    meta_data = pp.read_building_metadata(META_DATA_PATH)
    training_data = pp.read_data(TRAIN_DATA_PATH, TRAIN_WEATHER_PATH, meta_data)
    test_data = pp.read_data(TEST_DATA_PATH, TEST_WEATHER_PATH, meta_data)
    
    ## Prepare features and target
    features.prepare_features(training_data)
    target.get_log_target(training_data)
    features.prepare_features(test_data)
    
    ## Make data by meter type
    train_sets = splits.split_data_by_meter(training_data)
    test_sets = splits.split_data_by_meter(test_data)
    
    ## 
    