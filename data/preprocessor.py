import numpy as np
import pandas as pd
from utils import create_sequences, create_sequences_multistep, MinMaxScaler
from config import ModelConfig


class Preprocessor:
    def __init__(self, config = None):
        self.config = config or ModelConfig()
        self.scaler = MinMaxScaler()

    # == Training, Validation and Test Split (70-20-10) ==
    @staticmethod
    def train_val_test_split(series, train_ratio= 0.70, val_ratio= 0.20,):
        # 70-20-10 split of data
        n = len(series)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Di kasama dito yung actual testing on presentation
        return series.iloc[:train_end], series.iloc[train_end:val_end], series.iloc[val_end:]


    # == Build (X_train, y_train, X_val, y_val) in one call ==
    def prepare_multivariate(self, train_features, val_features, target, multistep = True,):
        # Fits the min-max scaler to the features
        scaled_train = self.scaler.fit_transform(train_features)

        # Scales the validation set to min-max (0-1)
        scaled_val = self.scaler.transform(val_features)
        self.scaler.save_scaler(target)

        # Produces forecast for output size > 1 (nakabase sa horizon)
        if multistep:
            X_train, y_train = create_sequences_multistep(scaled_train, self.config.lookback, self.config.horizon)
            X_val, y_val = create_sequences_multistep(scaled_val, self.config.lookback, self.config.horizon)

        # Produces forecast for output size = 1 (tomorrow lang ang prediction)    
        else:
            X_train, y_train = create_sequences(scaled_train, self.config.lookback)
            X_val, y_val = create_sequences(scaled_val, self.config.lookback)
            
        return X_train, y_train, X_val, y_val
