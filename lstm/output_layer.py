import numpy as np

class OutputLayer:
    # Ito yung y = W_y @ h + b_y.

    def __init__(self, hidden_size: int, output_size: int, scale: float, rng):
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_y = rng.uniform(-scale, scale, (output_size, hidden_size))
        self.b_y = np.zeros(output_size)

        self.reset_gradients()

    # == Similar to forward pass, updates yung weights and bias ==
    def forward(self, h: np.ndarray) -> np.ndarray:
        return self.W_y @ h + self.b_y

    # == Backward pjjass that updates yung gradients ==
    def backward(self, dy: np.ndarray, h: np.ndarray) -> np.ndarray:
        dh = self.W_y.T @ dy

        # Accumulate gradients
        self.dW_y += np.outer(dy, h)
        self.db_y += dy

        return dh

    def update_weights(self, learning_rate):
        self.W_y -= learning_rate * self.dW_y
        self.b_y -= learning_rate * self.db_y

        self.reset_gradients()

    def reset_gradients(self):
        self.dW_y = np.zeros_like(self.W_y)
        self.db_y = np.zeros_like(self.b_y)

    # == Get and set weights para sa update and restore ng  ModelCheckpoint ===
    def get_weights(self) -> dict:
        return {"W_y": self.W_y.copy(), "b_y": self.b_y.copy()}

    def set_weights(self, weights: dict):
        self.W_y = weights["W_y"].copy()
        self.b_y = weights["b_y"].copy()
