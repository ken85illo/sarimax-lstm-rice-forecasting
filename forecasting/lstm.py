import pandas as pd
from config import ModelConfig
from lstm.lstm_network import LSTMNetwork
from utils.min_max_scaler import MinMaxScaler
from utils.utils import split_by_chunks


class LSTM:
    def __init__(self, network, scaler: MinMaxScaler, config: ModelConfig):
        self.network = network
        self.scaler = scaler
        self.config = config

    # == Load LSTM with Config and Scaler ==
    @classmethod
    def load(LSTM, target, config: ModelConfig):
        network = LSTMNetwork(input_size=config.input_size, hidden_size=config.hidden_size, output_size=config.output_size)
        network.load_model(target)

        scaler = MinMaxScaler()
        scaler.load_scaler(target)

        return LSTM(network, scaler, config)

    # == Prediction ==
    def predict(self, residuals, trends, val_residuals_tail, val_trends_tail):
        # Input size * number of previous days to include 
        lookback = self.config.lookback

        # Number of days to forecast
        horizon = self.config.output_size

        # Prepend lookback rows from validation to seed the first window
        full_residuals = pd.concat([val_residuals_tail, residuals])
        full_trends = pd.concat([val_trends_tail, trends])

        feature_pairs = list(zip(full_residuals, full_trends))
        scaled = self.scaler.transform(feature_pairs)

        chunks = split_by_chunks(scaled, lookback)
        date_chunks = split_by_chunks(full_residuals.index, lookback)

        forecasts, dates = [], []
        for X_seq, date_window in zip(chunks, date_chunks):
            raw_pred = self.network.predict(X_seq).reshape(-1, 1)
            predicted = self.scaler.inverse_transform_feature(raw_pred, 0)

            self._print_window(X_seq, date_window, predicted, horizon)

            last_date = date_window[-1]
            for j in range(1, horizon + 1):
                forecast_date = last_date + pd.Timedelta(days=j)
                forecasts.extend(predicted[j - 1])
                dates.append(forecast_date)

        return pd.Series(forecasts, index=dates)

    # Pang print lang sa terminal
    def _print_window(self, X_seq, date_window, predicted, horizon):
        # Reverses (0-1) from min-max scaler back to normal rice price value
        inv = self.scaler.inverse_transform(X_seq)

        print(f"Date: {date_window[0].date()}\nInput:")
        for j, row in enumerate(inv):
            print(f"  {row} => {date_window[j].date()}")

        last_date = date_window[-1]
        print("Predicted:")
        for j in range(1, horizon + 1):
            fd = last_date + pd.Timedelta(days=j)
            print(f"  {predicted[j - 1]} => {fd.date()}")
            
        print()

