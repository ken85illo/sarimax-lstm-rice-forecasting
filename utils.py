import numpy as np

# Takes x and returns value between 0 and 1
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Takes x and returns value between -1 and 1
def tanh_function(x):
    return np.tanh(x)

# Derivative of tanh function
def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2

# Derivative of sigmoid function
def sigmoid_derivative(x):
    s = sigmoid_function(x)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_pred.shape[0]

def create_sequences(data, lookback):
    data = np.array(data)  # ← converts Series or list to numpy array

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def rmse(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred))
    
def mae(y_pred, y_true):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100    
