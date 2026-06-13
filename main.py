import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from utils import split_sentiment_classes
from config import ModelConfig, PathConfig
from data import DataLoader, Preprocessor
from lstm import LSTMNetwork
from training import Trainer
from forecasting import SARIMAX, LSTM, ResidualLearning

# == Shared config ==
CONFIG = {
    "high" : ModelConfig(
        lookback=14,
        horizon=1,
        hidden_size=16, # need to retrain if changed
        learning_rate=0.001,
        epochs=25,
        patience=10,
        input_size=5,
        output_size=1,
        dropout_rate=0.1
    ),
    "low": ModelConfig(
        lookback=14,
        horizon=14,
        hidden_size=32, # need to retrain if changed
        learning_rate=0.001,
        epochs=25,
        patience=10,
        input_size=5,
        output_size=1,
        dropout_rate=0.1

    )
}

PATHS = PathConfig()
loader = DataLoader(PATHS)

# == Helpers ==
def split_trends():
    # Return the google trends dataset with 2312 days (same size sa rice price)
    trends = loader.load_trends(max_rows=2312)
    return Preprocessor.train_val_test_split(trends, 0.70, 0.20)

def split_sentiments():
    sentiments = loader.load_sentiments(max_rows = 2312)
    return Preprocessor.train_val_test_split(sentiments, 0.70, 0.20)

# == Training pipelines ==
def train_lstm_residuals(target = "high"):
    print(f"=== Training: Well-Milled {target.capitalize()} ===")

    # Load raw data (rice price and google trends)
    train_resid = loader.load_train_residuals(target, f"Well-Milled {target.capitalize()}_train_residual") 
    val_resid = loader.load_val_residuals(target, f"Well-Milled {target.capitalize()}_val_residual")
    trends_train, trends_val, _ = split_trends()

    sentiment_train, sentiment_val, _ = split_sentiments()
    pos_train, neu_train, neg_train = split_sentiment_classes(sentiment_train)
    pos_val, neu_val, neg_val = split_sentiment_classes(sentiment_val)

    # Preprocess
    prep = Preprocessor(CONFIG[target])
    X_train, y_train, X_val, y_val = prep.prepare_training(
        train_features=list(zip(train_resid, trends_train, pos_train, neu_train, neg_train)),
        val_features=list(zip(val_resid, trends_val, pos_val, neu_val, neg_val)),
        target=target,
    )

    # Build, train, and save
    network = LSTMNetwork(
        input_size=CONFIG[target].input_size,
        hidden_size=CONFIG[target].hidden_size,
        output_size=CONFIG[target].output_size,
    )
    
    trainer = Trainer(network, CONFIG[target].learning_rate, CONFIG[target].epochs, CONFIG[target].patience, CONFIG[target].dropout_rate)
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.plot_predictions(
        target,
        f"Well-Milled {target.capitalize()} Residuals",
        prep.scaler,   
        X_train, y_train,
        X_val, y_val,
    )
    network.save_model(target)

def run_residual_forecast(rice_low_df, rice_high_df, enso_df, google_trends_df, sentiment_df):
    if (rice_low_df.index.min() != rice_high_df.index.min() or 
        rice_low_df.index.max() != rice_high_df.index.max()):
        return
    
    earliest_date = rice_low_df.index.min() 
    latest_date = rice_low_df.index.max()

    # SARIMAX models for Well-Milled Low and High
    sarimax_high = SARIMAX.load("high-test", PATHS)
    sarimax_low = SARIMAX.load("low-test", PATHS)

    # LSTM models for Well-Milled Low and High
    lstm_high = LSTM.load("high-test", CONFIG["high"], scaler_target="high") 
    lstm_low = LSTM.load("low-test", CONFIG["low"], scaler_target="low")

    # SARIMAX models for Well-Milled Low and High
    residual_learning_low = ResidualLearning(sarimax_low, lstm_low, "low", CONFIG["low"].horizon)
    residual_learning_high = ResidualLearning(sarimax_high, lstm_high, "high", CONFIG["high"].horizon)
    
    enso_original = loader.load_enso()
    
    last_enso = enso_df.index[0] - pd.Timedelta(days=CONFIG["high"].lookback)
    last_enso_df = enso_original[enso_original.index >= last_enso]
    extended_enso_df = pd.concat([last_enso_df, enso_df])
    
    # Test tails (last 14 days ng test to be passed as input)
    test_resid_high = loader.load_test_residuals("high")
    test_resid_low = loader.load_test_residuals("low")

    test_resid_high_tail = test_resid_high.iloc[-CONFIG["high"].lookback:]
    test_resid_low_tail = test_resid_low.iloc[-CONFIG["low"].lookback:]

    _, _ , trends_test = split_trends()
    _, _, sentiment_test = split_sentiments()

    test_trends_tail = trends_test.iloc[-CONFIG["high"].lookback:]
    test_sentiment_tail = sentiment_test.iloc[-CONFIG["high"].lookback:]
    
    residual_learning_low.run_rolling(
        endog=rice_low_df,
        exog=extended_enso_df,
        trends_test=google_trends_df,
        sentiment_test=sentiment_df,
        residuals_tail=test_resid_low_tail,
        trends_tail=test_trends_tail,
        sentiment_tail=test_sentiment_tail,
        start_date=earliest_date,
        end_date=latest_date,
        target_csv="low",
        is_test_set=False,
        target='low'
    )

    residual_learning_high.run_rolling(
        endog=rice_high_df,
        exog=extended_enso_df,
        trends_test=google_trends_df,
        sentiment_test=sentiment_df,
        residuals_tail=test_resid_high_tail,
        trends_tail=test_trends_tail,
        sentiment_tail=test_sentiment_tail,
        start_date=earliest_date,
        end_date=latest_date,
        target_csv='high',
        is_test_set=False,
        target='high'
    )

# == Residual Learning for Test Set ==
def run_residual_learning_test_set(target = "high"):
    print(f"=== Residual Learning Evaluation: Well-Milled {target.capitalize()} ===")

    rice_df = loader.load_rice()
    enso = loader.load_enso()
    _, trends_val, trends_test = split_trends()
    _, sentiment_val, sentiment_test = split_sentiments()

    # Test window
    test_start = pd.to_datetime("2025-09-11")
    test_end = rice_df[f"Well-Milled {target.capitalize()}"].index.max()

    endog_test = rice_df[f"Well-Milled {target.capitalize()}"][rice_df.index >= "2025-08-28"]
    exog_test = enso[enso.index >= "2025-08-28"]

    # Load models
    sarimax = SARIMAX.load(target, PATHS)
    lstm = LSTM.load(target, CONFIG[target])

    # Validation tails (last 14 days ng validation to be passed as input)
    val_resid = loader.load_val_residuals(target, f"Well-Milled {target.capitalize()}_val_residual")
    val_resid_tail = val_resid.iloc[-CONFIG[target].lookback:]
    val_trends_tail = trends_val.iloc[-CONFIG[target].lookback:]
    val_sentiment_tail = sentiment_val.iloc[-CONFIG[target].lookback:]

    residual_learning = ResidualLearning(sarimax, lstm, target=target, steps=CONFIG[target].horizon)
    residual_learning.run_rolling(
        endog=endog_test,
        exog=exog_test,
        trends_test=trends_test,
        sentiment_test=sentiment_test,
        residuals_tail=val_resid_tail,
        trends_tail=val_trends_tail,
        sentiment_tail=val_sentiment_tail,
        start_date=test_start,
        end_date=test_end,
        target_csv=target,
        is_test_set=True,
        target=target
    )

    # Used just to save the model after test set
    sarimax.save_model(f"{target}-test")
    lstm.network.save_model(f"{target}-test")

# == Sanity Check == 
def sanity_check_overfit():
    print("=== Sanity check (overfitting a single sample) ===")

    network = LSTMNetwork(input_size=3, hidden_size=64, output_size=1)
    trainer = Trainer(network, learning_rate=0.1, epochs=200, patience=20)

    X_sample = np.random.randn(5, 3)
    y_sample = np.array([0.8])

    trainer.train([X_sample], [y_sample], [X_sample], [y_sample])

    pred = network.predict(X_sample)
    print(f"\nTarget: {y_sample}  |  Prediction: {pred}")

# == Entry point ==
if __name__ == "__main__":
    train_lstm_residuals(target="low")
    # run_residual_learning_test_set(target="high")
    run_residual_learning_test_set(target="low")
