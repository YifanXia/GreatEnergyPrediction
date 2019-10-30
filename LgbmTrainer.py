import lightgbm as lgbm
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder, OrdinalEncoder
#from sklearn.preprocessing import OrdinalEncoder
from typing import Dict, Tuple
from splits import split_train_validation
from params import FEATURE_LIST, CATEGORICAL_LIST, TARGET_COL
import pickle

DEFAULT_PARAMS = {
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_iterations': 3000,
    'max_depth': 4,
    'objective': 'regression',
    'min_child_samples': 10,
    'subsample': .9,
    'colsample_bytree': .8,
    'metric_freq': 100,
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
        self.encoder = None
    
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
        self.encoder = OrdinalEncoder()#TargetEncoder()
        self.encoder.fit(features[CATEGORICAL_LIST])#, target)
        
    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features.loc[:, CATEGORICAL_LIST] = self.encoder.transform(features[CATEGORICAL_LIST])
        return features

    def train(self, data: pd.DataFrame, params: Dict = None) -> None:
        if params is None:
            params = {}
        fit_params = {**FIT_PARAMS, **params}
        best_params = self._train(data, fit_params)
        train_features, train_target = self.get_features_target(data)
        self.get_categorical_encoder(train_features, train_target)
        train_features = self.encode(train_features)
        fit_params.pop('early_stopping_rounds', None)
        self.model.set_params(num_iterations=self.model.best_iteration_)
        self.model.fit(train_features, train_target, categorical_feature=CATEGORICAL_LIST, **fit_params)
        
    def _train(self, data: pd.DataFrame, params: Dict = None) -> Dict:
        if params is None:
            params = {}
        fit_params = {**FIT_PARAMS, **params}
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
    