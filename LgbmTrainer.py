import lightgbm as lgbm
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from splits import split_train_validation, time_based_split_train_validation
from params import FEATURE_LIST, CATEGORICAL_LIST, TARGET_COL
import pickle

DEFAULT_PARAMS = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_iterations': 3000,
    'max_depth': 6,
    'objective': 'regression',
    'min_child_samples': 10,
    'num_leaves': 32,
    #'subsample': .9,
    'subsample': 0.4,
    'colsample_bytree': .8,
    'metric_freq': 100,
    'silent': False,
}

FIT_PARAMS = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 100, 
    'verbose': 100,
    'feature_name': 'auto',
}

class LgbmModel:
    
    def __init__(self, params: Dict = None):
        if params is None:
            params = {}
        self.model = lgbm.LGBMRegressor(**{**DEFAULT_PARAMS, **params})
        self.encoder = {}
    
    def read_model(self, file_name: str):
        with open(file_name, 'rb') as fin:
            model, encoder = pickle.load(fin)
        self.model = model
        self.encoder = encoder
    
    @staticmethod
    def get_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        features = data[FEATURE_LIST]
        target = data[TARGET_COL]
        return (features, target)
    
    def get_categorical_encoder(self, features: pd.DataFrame, target: pd.DataFrame):
        self.encoder = {}
        logging.info(f'Categorical features are {CATEGORICAL_LIST}.')
        for col in CATEGORICAL_LIST:
            col_type = features[col].dtypes
            if col_type == 'object':
                logging.info(f'Encoding {col}')
                self.encoder[col] = {k: v for v, k in enumerate(features[col].unique())}
        logging.info('Finished categorical encoder')
        
    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        logging.info('Transform categorical features.')
        for col in CATEGORICAL_LIST:
            if col in self.encoder:
                logging.info(f'Transforming {col}')
                features.loc[:, col] = features[col].map(self.encoder[col])
        features.loc[:, CATEGORICAL_LIST] = features[CATEGORICAL_LIST].astype('category')
        return features

    def train(self, data: pd.DataFrame, params: Dict = None,
              use_time_based_split: bool = False, valid_size_days: int = 120) -> None:
        if params is None:
            params = {}
        fit_params = {**FIT_PARAMS, **params}
        best_params = self._train(data, fit_params, use_time_based_split=use_time_based_split)
        logging.info('Retraining using the entire dataset...')
        train_features, train_target = self.get_features_target(data)
        self.get_categorical_encoder(train_features, train_target)
        train_features = self.encode(train_features)
        fit_params.pop('early_stopping_rounds', None)
        self.model.set_params(num_iterations=self.model.best_iteration_)
        logging.info(f'Number of iterations is set to {self.model.get_params()["num_iterations"]}')
        self.model.fit(train_features, train_target, categorical_feature=CATEGORICAL_LIST, **fit_params)
        
    def _train(self, data: pd.DataFrame, params: Dict = None,
               use_time_based_split: bool = False, valid_size_days: int = 120) -> Dict:
        if params is None:
            params = {}
        fit_params = {**FIT_PARAMS, **params}
        if use_time_based_split and valid_size_days:
            logging.info('Use time-based train-validation split.')
            logging.info(f'Validation set contains {valid_size_days} days.')
            train_data, valid_data = time_based_split_train_validation(data, valid_size_days)
        else:
            logging.info('Use random train-validation split.')
            train_data, valid_data = split_train_validation(data)
        train_features, train_target = self.get_features_target(train_data)
        valid_features, valid_target = self.get_features_target(valid_data)
        self.get_categorical_encoder(train_features, train_target)
        train_features = self.encode(train_features)
        valid_features = self.encode(valid_features)
        self.model.fit(train_features, train_target, eval_set=[(valid_features, valid_target)],
                       categorical_feature=CATEGORICAL_LIST, **fit_params)
        
        return self.model.get_params()
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        features = data[FEATURE_LIST]
        features = self.encode(features)
        #breakpoint()
        return self.model.predict(features)
    
    def save_model(self, file_name: str) -> None:
        with open(file_name, 'wb') as fout:
            pickle.dump((self.model, self.encoder), fout)
    