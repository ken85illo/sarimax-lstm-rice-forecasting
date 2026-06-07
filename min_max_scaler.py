import numpy as np

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        data = np.array(data)
        # Calculate min and max per feature
        self.min = np.min(data, axis=0) 
        self.max = np.max(data, axis=0)

    def transform(self, data):
        data = np.array(data)

        # Apply the formula: (x - min) / (max - min)
        return (data - self.min) / (self.max - self.min)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        scaled_data = np.array(scaled_data)

        # Useful for converting predictions back to original price scale
        return scaled_data * (self.max - self.min) + self.min

    def save_scaler(self, target):
        filename = f"{target}-scaler.npz"
        
        # We save all learnable parameters into a single .npz file
        np.savez(filename, min = self.min, max = self.max)
        print(f"Scaler saved to {filename}")

    def load_scaler(self, target):
        filename = f"{target}-scaler.npz"

        data = np.load(filename)
        self.min = data['min']
        self.max = data['max']

        print(f"Scaler loaded from {filename}")
