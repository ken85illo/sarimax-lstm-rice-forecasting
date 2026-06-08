import numpy as np
from lstm import LSTMCell
from lstm import OutputLayer

class LSTMNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm_cell_1 = LSTMCell(input_size, hidden_size)
        self.lstm_cell_2 = LSTMCell(hidden_size, hidden_size)

        self.output_layer = OutputLayer(
            hidden_size=hidden_size,
            output_size=output_size,
            scale=self.lstm_cell_2.scale,
            rng=self.lstm_cell_2.rng,
        )

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        out = self.output_layer
        h1, c1 = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        h2, c2 = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        
        for t in range(len(X_seq)):
            h1, c1 = self.lstm_cell_1.forward_pass(X_seq[t], h1, c1)
            h2, c2 = self.lstm_cell_2.forward_pass(h1, h2, c2)

        output = out.W_y @ h2 + out.b_y
        return output

    # == Saving model to avoid retraining everytime ==
    def save_model(self, target: str):
        filename = f"checkpoint/{target}-lstm_model.npz"
        out = self.output_layer

        cell_1 = self.lstm_cell_1
        cell_2 = self.lstm_cell_2

        # Saves as .npz na file
        np.savez(
            filename,
            # Layer 1
            c1_W_f=cell_1.W_f, c1_b_f=cell_1.b_f,
            c1_W_i=cell_1.W_i, c1_b_i=cell_1.b_i,
            c1_W_c=cell_1.W_c, c1_b_c=cell_1.b_c,
            c1_W_o=cell_1.W_o, c1_b_o=cell_1.b_o,
            # Layer 2
            c2_W_f=cell_2.W_f, c2_b_f=cell_2.b_f,
            c2_W_i=cell_2.W_i, c2_b_i=cell_2.b_i,
            c2_W_c=cell_2.W_c, c2_b_c=cell_2.b_c,
            c2_W_o=cell_2.W_o, c2_b_o=cell_2.b_o,
            # Output
            W_y=out.W_y, b_y=out.b_y
        )


        print(f"Model saved to {filename}")

    # == Load model for use (.npz) ==
    def load_model(self, target: str):
        filename = f"checkpoint/{target}-lstm_model.npz"

        data = np.load(filename)
        out = self.output_layer

        cell_1 = self.lstm_cell_1
        cell_2 = self.lstm_cell_2

        cell_1.W_f, cell_1.b_f = data['c1_W_f'], data['c1_b_f']
        cell_1.W_i, cell_1.b_i = data['c1_W_i'], data['c1_b_i']
        cell_1.W_c, cell_1.b_c = data['c1_W_c'], data['c1_b_c']
        cell_1.W_o, cell_1.b_o = data['c1_W_o'], data['c1_b_o']

        cell_2.W_f, cell_2.b_f = data['c2_W_f'], data['c2_b_f']
        cell_2.W_i, cell_2.b_i = data['c2_W_i'], data['c2_b_i']
        cell_2.W_c, cell_2.b_c = data['c2_W_c'], data['c2_b_c']
        cell_2.W_o, cell_2.b_o = data['c2_W_o'], data['c2_b_o']

        out.W_y, out.b_y = data['W_y'], data['b_y']

        print(f"Model loaded from {filename}")