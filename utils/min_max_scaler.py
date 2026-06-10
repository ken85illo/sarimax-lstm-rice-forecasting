import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    # Set the min and max of scaler
    def fit(self, data):
        data = np.array(data)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return self

    # Transform the features
    def transform(self, data):
        data = np.array(data)
        return (data - self.min) / (self.max - self.min)

    # Fit and transform the features
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    # Revert the transformed features
    def inverse_transform(self, scaled_data):
        scaled_data = np.array(scaled_data)
        return scaled_data * (self.max - self.min) + self.min

    # Revert a single transformed feature
    def inverse_transform_feature(self, scaled_data, feature_idx: int):
        # Inverse transform a single feature
        scaled_data = np.array(scaled_data)
        min = self.min[feature_idx]
        max = self.max[feature_idx]
        return scaled_data * (max - min) + min

    def save_scaler(self, target: str):
        filename = f"checkpoint/{target}-scaler.npz"
        np.savez(filename, min=self.min, max=self.max)
        print(f"Scaler saved to {filename}")

    def load_scaler(self, target: str):
        filename = f"checkpoint/{target}-scaler.npz"
        data = np.load(filename)
        self.min = data["min"]
        self.max = data["max"]
        print(f"Scaler loaded from {filename}")
