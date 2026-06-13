import numpy as np
from lstm import LSTMCell
from lstm import OutputLayer

class LSTMNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.forward_cell = LSTMCell(input_size, hidden_size)
        self.backward_cell = LSTMCell(input_size, hidden_size)

        self.output_layer = OutputLayer(
            hidden_size=hidden_size * 2,
            output_size=output_size,
        )

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        T = len(X_seq)

        h_fw = np.zeros(self.hidden_size)
        c_fw = np.zeros(self.hidden_size)
        h_bw = np.zeros(self.hidden_size)
        c_bw = np.zeros(self.hidden_size)

        fw_hs = []
        bw_hs = [None] * T

        for t in range(T):
            h_fw, c_fw, _ = self.forward_cell.forward_pass(X_seq[t], h_fw, c_fw)
            fw_hs.append(h_fw)

        for t in reversed(range(T)):
            h_bw, c_bw, _ = self.backward_cell.forward_pass(X_seq[t], h_bw, c_bw)
            bw_hs[t] = h_bw

        h_concat_final = np.concatenate((fw_hs[-1], bw_hs[0]), axis=0)
        predictions = self.output_layer.forward(h_concat_final)

        return predictions.reshape(-1, 1)

    # == Saving model to avoid retraining everytime ==
    def save_model(self, target: str):
        filename = f"checkpoint/{target}-lstm_model.npz"
        fw = self.forward_cell
        bw = self.backward_cell
        out = self.output_layer

        # Saves as .npz na file
        np.savez(
            filename,
            W_f_fw=fw.W_f, b_f_fw=fw.b_f,
            W_i_fw=fw.W_i, b_i_fw=fw.b_i,
            W_c_fw=fw.W_c, b_c_fw=fw.b_c,
            W_o_fw=fw.W_o, b_o_fw=fw.b_o,
            
            W_f_bw=bw.W_f, b_f_bw=bw.b_f,
            W_i_bw=bw.W_i, b_i_bw=bw.b_i,
            W_c_bw=bw.W_c, b_c_bw=bw.b_c,
            W_o_bw=bw.W_o, b_o_bw=bw.b_o,
            
            W_y=out.W_y,  b_y=out.b_y,
        )

        print(f"Model saved to {filename}")

    # == Load model for use (.npz) ==
    def load_model(self, target: str):
        filename = f"checkpoint/{target}-lstm_model.npz"

        data = np.load(filename)
        fw = self.forward_cell
        bw = self.backward_cell
        out = self.output_layer

        fw.W_f, fw.b_f = data["W_f_fw"], data["b_f_fw"]
        fw.W_i, fw.b_i = data["W_i_fw"], data["b_i_fw"]
        fw.W_c, fw.b_c = data["W_c_fw"], data["b_c_fw"]
        fw.W_o, fw.b_o = data["W_o_fw"], data["b_o_fw"]

        bw.W_f, bw.b_f = data["W_f_bw"], data["b_f_bw"]
        bw.W_i, bw.b_i = data["W_i_bw"], data["b_i_bw"]
        bw.W_c, bw.b_c = data["W_c_bw"], data["b_c_bw"]
        bw.W_o, bw.b_o = data["W_o_bw"], data["b_o_bw"]

        out.W_y, out.b_y = data["W_y"], data["b_y"]

        print(f"Model loaded from {filename}")
