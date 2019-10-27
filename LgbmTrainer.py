import lightgbm as lgbm
import pandas as pd
from category_encoders import TargetEncoder
from typing import Dict, Tuple
from params import FEATURE_LIST, CATEGORICAL_LIST, TARGET_COL

DEFAULT_PARAMS = {}

FIT_PARAMS = {}

class LgbmModel:
    
    def __init__(self, params: Dict = None):
        if params is None:
            params = {}
        self.model = lgbm.LGBMRegressor(**{**DEFAULT_PARAMS, **params})
        self.encoder = None
        
    @staticmethod
    def get_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        features = data[FEATURE_LIST]
        target = data[TARGET_COL]
        return (features, target)
    
    def get_categorical_encoder(self, features: pd.DataFrame, target: pd.DataFrame):
        self.encoder = TargetEncoder()
        self.encoder.fit(features[CATEGORICAL_LIST], target)
        
    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        features.loc[:, CATEGORICAL_LIST] = self.encoder.transform(features[CATEGORICAL_LIST])
        return features

    def train(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, params: Dict = None):
        if params is None:
            params = {}
        fit_params = {**FIT_PARAMS, **params}
        train_features, train_target = self.get_features_target(train_data)
        valid_features, valid_target = self.get_features_target(valid_data)
        self.get_categorical_encoder(train_features, train_target)
        train_features = self.encode(train_features)
        valid_features = self.encode(valid_features)
        self.model.fit(train_features, train_target, eval_set=[(valid_features, valid_target)], **fit_params)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        features = data[FEATURE_LIST]
        features = self.encode(features)
        return self.model.predict(features)
    