import numpy as np


# == Activation functions for LSTM ==

def sigmoid_function(x):
    # Maps x to (0, 1)
    return 1 / (1 + np.exp(-x))

def tanh_function(x):
    # Maps x to (-1, 1)
    return np.tanh(x)

def tanh_derivative(x):
    # Derivative of tanh given its output value
    return 1 - x ** 2


def sigmoid_derivative(x):
    # Derivative of sigmoid given its output value
    return x * (1 - x)


# == Loss functions ==
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_pred.shape[0]


# == Evaluation metrics ==
def mae(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# == Create Sequences Helpers (ginagamit sa LSTM training) ===
def create_sequences(data, lookback):
    # Creates X (input) and y (target) pairs
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def create_sequences_multistep(data, lookback, horizon):
    # Create (X, y) pairs for multi-step prediction
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i:i + horizon, 0])

    return np.array(X), np.array(y)


def split_by_chunks(data, chunk_size):
    # Split an array into non-overlapping chunks of chunk_size (for LSTM prediction)
    return [data[i:i + chunk_size] for i in range(0, len(data) - chunk_size + 1, chunk_size)]

def split_sentiment_classes(sentiment_df):
    return (sentiment_df[c] for c in ['score_positive', 'score_neutral', 'score_negative'])
