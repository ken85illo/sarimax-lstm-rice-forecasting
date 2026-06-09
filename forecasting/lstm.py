import pandas as pd
from config import ModelConfig
from lstm.lstm_network import LSTMNetwork
from utils.min_max_scaler import MinMaxScaler
from utils.utils import print_tabulation, split_by_chunks, split_sentiment_classes


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
    def predict(self, residual_window, trends_window, sentiment_window, current_date):
        pos_window, neu_window, neg_window = split_sentiment_classes(sentiment_window)
        feature_pairs = list(zip(residual_window, trends_window, pos_window, neu_window, neg_window))
        scaled_input = self.scaler.transform(feature_pairs)

        raw_pred = self.network.predict(scaled_input).reshape(-1, 1)
        predicted = self.scaler.inverse_transform_feature(raw_pred, 0)

        self._print_window(scaled_input, predicted, current_date)
        return predicted

    # Pang print lang sa terminal
    def _print_window(self, scaled_input, predicted, current_date):
        # Reverses (0-1) from min-max scaler back to normal rice price value
        inverse_input = self.scaler.inverse_transform(scaled_input)
        current_date = current_date.date()
        window_start = current_date - pd.Timedelta(days=len(predicted))

        input_dates = [window_start + pd.Timedelta(days=t) for t in range(len(inverse_input)) ]
        output_dates = [current_date + pd.Timedelta(days=t) for t in range(len(predicted)) ]

        print_df = pd.DataFrame({
            "Input": input_dates,
            "Residual": inverse_input[:, 0],
            "Google Trends": inverse_input[:, 1],
            "Positive": inverse_input[:, 2],
            "Neutral": inverse_input[:, 3],
            "Negative": inverse_input[:, 3],
            "Output": output_dates,
            "Prediction": predicted.flatten(),
        })
        print()
        print_tabulation(print_df, title="=== LSTM CORRECTION ===")


