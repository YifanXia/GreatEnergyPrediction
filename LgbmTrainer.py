import lightgbm as lgbm
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List
from splits import (
    split_train_validation, 
    time_based_split_train_validation, 
    time_based_half_year_split,
    odd_even_month_half_year_split
    )
from params import FEATURE_LIST, CATEGORICAL_LIST, TARGET_COL
import pickle
import matplotlib.pyplot as plt

DFList = List[pd.DataFrame]

DEFAULT_PARAMS = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_iterations': 3000,
    #'max_depth': 4,
    'objective': 'regression',
    'min_child_samples': 20,
    'num_leaves': 40,
    #'subsample': .9,
    'subsample': 0.8,
    'colsample_bytree': .9,
    'metric_freq': 100,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
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
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = []
        self.encoder = []
    
    def read_model(self, file_name: str):
        with open(file_name, 'rb') as fin:
            model, encoder = pickle.load(fin)
        self.model = model
        self.encoder = encoder
    
    @staticmethod
    def get_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.debug(f'Features are {FEATURE_LIST}')
        features = data[FEATURE_LIST]
        target = data[TARGET_COL]
        return (features, target)
    
    @staticmethod
    def get_categorical_encoder(features: pd.DataFrame, target: pd.DataFrame) -> Dict:
        encoder = {}
        logging.info(f'Categorical features are {CATEGORICAL_LIST}.')
        for col in CATEGORICAL_LIST:
            col_type = features[col].dtypes
            if col_type == 'object':
                logging.info(f'Encoding {col}')
                encoder[col] = {k: v for v, k in enumerate(features[col].unique())}
        logging.info('Finished categorical encoder')
        return encoder
    
    @staticmethod
    def encode(encoder: Dict, features: pd.DataFrame) -> pd.DataFrame:
        logging.info('Transform categorical features.')
        for col in CATEGORICAL_LIST:
            if col in encoder:
                logging.info(f'Transforming {col}')
                features.loc[:, col] = features[col].map(encoder[col])
                logging.debug(f'Encoded {col} are: {features.loc[:, col].unique()}')
        features.loc[:, CATEGORICAL_LIST] = features[CATEGORICAL_LIST].astype('category')
        return features

    def train(self, data: pd.DataFrame, params: Dict = None) -> None:
        if params is None:
            params = {}
        first_half, second_half = time_based_half_year_split(data)#odd_even_month_half_year_split(data)
        logging.info('Training on the 1st half, validation on the 2nd')
        self._train([first_half, second_half])
        logging.info('Training on the 2nd half, validation on the 1st')
        self._train([second_half, first_half])
        
    def _train(self, data: DFList, params: Dict = None) -> None:
        if params is None:
            params = {}
        fit_params = {**FIT_PARAMS, **params}
        train_data = data[0]
        valid_data = data[1]
        train_features, train_target = self.get_features_target(train_data)
        valid_features, valid_target = self.get_features_target(valid_data)
        encoder = self.get_categorical_encoder(train_features, train_target)
        train_features = self.encode(encoder, train_features)
        valid_features = self.encode(encoder, valid_features)
        model = lgbm.LGBMRegressor(**self.params)
        model.fit(train_features, train_target, eval_set=[(valid_features, valid_target)],
                  categorical_feature=CATEGORICAL_LIST, **fit_params)
        self.model.append(model)
        self.encoder.append(encoder)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        prediction = np.zeros(data.shape[0])
        for i in range(len(self.model)):
            features = data[FEATURE_LIST]
            features = self.encode(self.encoder[i], features)
            prediction += np.clip(np.expm1(self.model[i].predict(features)), 0, None)
        return prediction / len(self.model)
    
    def plot_feature_importances(self, file_name: str) -> None:
        f_imp_1 = self.model[0].booster_.feature_importance(importance_type='gain')
        f_imp_2 = self.model[1].booster_.feature_importance(importance_type='gain')
        feature_importances = [
            (FEATURE_LIST[index], v1 + v2) for index, v1, v2 in zip(list(range(len(FEATURE_LIST))), f_imp_1, f_imp_2)
        ]
        feature_importances = list(
            map(list, zip(*sorted(feature_importances, key=lambda x: x[1], reverse=True)))
        )
        plt.figure(figsize=(len(FEATURE_LIST) // 2, 5))
        plt.bar(feature_importances[0], feature_importances[1])
        plt.xticks(rotation='vertical')
        plt.savefig(file_name, format='png')
        
    def save_model(self, file_name: str) -> None:
        with open(file_name, 'wb') as fout:
            pickle.dump((self.model, self.encoder), fout)
    