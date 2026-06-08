import numpy as np
from utils import mse_loss, mse_loss_derivative


class OutputLayer:
    # Ito yung y = W_y @ h + b_y.

    def __init__(self, hidden_size: int, output_size: int, scale: float, rng):
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_y = scale * rng.normal(0.0, 0.1, (output_size, hidden_size))
        self.b_y = np.zeros(output_size)

    # == Similar to forward pass, updates yung weights and bias ==
    def forward(self, h: np.ndarray) -> np.ndarray:
        return self.W_y @ h + self.b_y

    # == MSE Loss functions ==
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return mse_loss(y_pred, y_true)

    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return mse_loss_derivative(y_pred, y_true)

    # == Backward pjjass that updates yung gradients ==
    def backward(self, dy: np.ndarray, h: np.ndarray, learning_rate: float) -> np.ndarray:
        dh = self.W_y.T @ dy

        self.W_y -= learning_rate * np.outer(dy, h)
        self.b_y -= learning_rate * dy

        return dh

    # == Get and set weights para sa update and restore ng  ModelCheckpoint ===
    def get_weights(self) -> dict:
        return {"W_y": self.W_y.copy(), "b_y": self.b_y.copy()}

    def set_weights(self, weights: dict):
        self.W_y = weights["W_y"]
        self.b_y = weights["b_y"]