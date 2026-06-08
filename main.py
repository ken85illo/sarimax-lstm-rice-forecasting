import numpy as np
import pandas as pd

from utils import split_sentiment_classes
from config import ModelConfig, PathConfig
from data import DataLoader, Preprocessor
from lstm import LSTMNetwork
from training import Trainer
from forecasting import SARIMAX, LSTM, ResidualLearning

# == Shared config ==
CONFIG = ModelConfig(
    lookback=14,
    horizon=14,
    hidden_size=64,
    learning_rate=0.001,
    epochs=300,
    patience=20,
    input_size=5,
    output_size=14,
)

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
    prep = Preprocessor(CONFIG)
    X_train, y_train, X_val, y_val = prep.prepare_multivariate(
        train_features=list(zip(train_resid, trends_train, pos_train, neu_train, neg_train)),
        val_features=list(zip(val_resid, trends_val, pos_val, neu_val, neg_val)),
        target=target,
        multistep=True,
    )

    # Build, train, and save
    network = LSTMNetwork(
        input_size=CONFIG.input_size,
        hidden_size=CONFIG.hidden_size,
        output_size=CONFIG.output_size,
    )
    
    trainer = Trainer(network, CONFIG.learning_rate, CONFIG.epochs, CONFIG.patience)
    trainer.train(X_train, y_train, X_val, y_val)
    network.save_model(target)



# == Residual Learning Evaluation ==
def run_residual_learning_evaluation(target = "high"):
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
    lstm = LSTM.load(target, CONFIG)

    # Validation tails (last 14 days ng validation to be passed as input)
    val_resid = loader.load_val_residuals(target, f"Well-Milled {target.capitalize()}_val_residual")
    val_resid_tail = val_resid.iloc[-CONFIG.lookback:]
    val_trends_tail = trends_val.iloc[-CONFIG.lookback:]
    val_sentiment_tail = sentiment_val.iloc[-CONFIG.lookback:]


    residual_learning = ResidualLearning(sarimax, lstm)
    residual_learning.run(
        endog=endog_test,
        exog=exog_test,
        trends_test=trends_test,
        sentiment_test=sentiment_test,
        val_residuals_tail=val_resid_tail,
        val_trends_tail=val_trends_tail,
        val_sentiment_tail=val_sentiment_tail,
        start_date=test_start,
        end_date=test_end,
        steps=CONFIG.horizon,
        output_csv=f"output/final_forecast_{target}.csv",
    )

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
    train_lstm_residuals(target="high")
    run_residual_learning_evaluation(target="high")