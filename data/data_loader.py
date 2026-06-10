import pandas as pd
from config import PathConfig

class DataLoader:
    def __init__(self, paths = None):
        self.paths = paths or PathConfig()

    # == Loads Rice prices CSV ==
    def load_rice(self):
        df = pd.read_csv(self.paths.rice_csv, parse_dates=["Date"], index_col="Date")
        df = df.sort_index()

        return df

    # == Loads ENSO Index CSV ==
    def load_enso(self):
        df = pd.read_csv(self.paths.enso_csv, parse_dates=["date"], index_col="date")

        return df["ONI value"]

    # == Loads Google Trends CSV ==
    def load_trends(self, max_rows):
        df = pd.read_csv(self.paths.combined_trends_csv)
        df = df[:max_rows]
        df["Day"] = pd.to_datetime(df["Day"])
        df = df.set_index("Day")

        return df["Google Trends"]

    def load_sentiments(self, max_rows):
        df = pd.read_csv(self.paths.sentiments_csv)
        df = df[:max_rows]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        
        return df

    # == Loads SARIMAX Residuals CSV ==
    def load_train_residuals(self, label, col):
        path = self.paths.train_residuals(label)
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        return df[col]

    def load_val_residuals(self, label, col):
        path = self.paths.val_residuals(label)
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        return df[col]
    
    def load_test_residuals(self, label):
        path = self.paths.test_residuals(label)
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        return df["Residuals"]
