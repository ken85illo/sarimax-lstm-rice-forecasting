import numpy as np
from lstm import LSTMCell
from lstm import OutputLayer

class LSTMNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.output_layer = OutputLayer(
            hidden_size=hidden_size,
            output_size=output_size,
        )

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        T = len(X_seq)

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        for t in range(T):
            h, c, _ = self.lstm_cell.forward_pass(X_seq[t], h, c)

        predictions = self.output_layer.forward(h)
        return predictions.reshape(-1, 1)

    def predict_multistep(self, X_seq, steps=14):
        predictions = []
        current_seq = X_seq.copy()
        
        for _ in range(steps):
            pred = self.predict(current_seq)  # predict 1 step
            predictions.append(pred[0])
            
            # Shift window and append prediction as next input
            new_step = current_seq[-1].copy()
            new_step[0] = np.array(pred.flatten())[0]  # update residual feature
            current_seq = np.vstack([current_seq[1:], new_step])
        
        return np.array(predictions)

    # == Saving model to avoid retraining everytime ==
    def save_model(self, target: str):
        filename = f"checkpoint/{target}-lstm_model.npz"
        cell = self.lstm_cell
        out = self.output_layer

        # Saves as .npz na file
        np.savez(
            filename,
            W_f=cell.W_f, b_f=cell.b_f,
            W_i=cell.W_i, b_i=cell.b_i,
            W_c=cell.W_c, b_c=cell.b_c,
            W_o=cell.W_o, b_o=cell.b_o,
            W_y=out.W_y, b_y=out.b_y,
        )
        print(f"Model saved to {filename}")

    # == Load model for use (.npz) ==
    def load_model(self, target: str):
        filename = f"checkpoint/{target}-lstm_model.npz"

        data = np.load(filename)
        cell = self.lstm_cell
        out = self.output_layer

        cell.W_f, cell.b_f = data["W_f"], data["b_f"]
        cell.W_i, cell.b_i = data["W_i"], data["b_i"]
        cell.W_c, cell.b_c = data["W_c"], data["b_c"]
        cell.W_o, cell.b_o = data["W_o"], data["b_o"]

        out.W_y, out.b_y = data["W_y"], data["b_y"]

        print(f"Model loaded from {filename}")
