from lstm.lstm_network import LSTMNetwork


class ModelCheckpoint:
    def __init__(self):
        self.best_loss= float("inf")
        self._weights = None

        # Observes change up till 6th decimal place
        self.min_delta = 1e-6

    # == Updates the network weights ==
    def update(self, network, val_loss):
        # Cheks for valid loss
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self._weights = self._snapshot(network)
            
            return True

        return False

    # == Loads the best saved weights ==
    def restore(self, network):
        if self._weights is None:
            return
        
        # Restore Forward Cell
        fw = network.forward_cell
        fw.W_f, fw.b_f = self._weights["W_f_fw"].copy(), self._weights["b_f_fw"].copy()
        fw.W_i, fw.b_i = self._weights["W_i_fw"].copy(), self._weights["b_i_fw"].copy()
        fw.W_c, fw.b_c = self._weights["W_c_fw"].copy(), self._weights["b_c_fw"].copy()
        fw.W_o, fw.b_o = self._weights["W_o_fw"].copy(), self._weights["b_o_fw"].copy()

        # Restore Backward Cell
        bw = network.backward_cell
        bw.W_f, bw.b_f = self._weights["W_f_bw"].copy(), self._weights["b_f_bw"].copy()
        bw.W_i, bw.b_i = self._weights["W_i_bw"].copy(), self._weights["b_i_bw"].copy()
        bw.W_c, bw.b_c = self._weights["W_c_bw"].copy(), self._weights["b_c_bw"].copy()
        bw.W_o, bw.b_o = self._weights["W_o_bw"].copy(), self._weights["b_o_bw"].copy()

        network.output_layer.set_weights(
            {"W_y": self._weights["W_y"].copy(), "b_y": self._weights["b_y"].copy()}
        )

    # == Returns the current weights and biases ng cell == 
    @staticmethod
    def _snapshot(network):
        fw = network.forward_cell
        bw = network.backward_cell
        out_weights = network.output_layer.get_weights()

        return {
            # Forward weights
            "W_f_fw": fw.W_f.copy(), "b_f_fw": fw.b_f.copy(),
            "W_i_fw": fw.W_i.copy(), "b_i_fw": fw.b_i.copy(),
            "W_c_fw": fw.W_c.copy(), "b_c_fw": fw.b_c.copy(),
            "W_o_fw": fw.W_o.copy(), "b_o_fw": fw.b_o.copy(),
            # Backward weights
            "W_f_bw": bw.W_f.copy(), "b_f_bw": bw.b_f.copy(),
            "W_i_bw": bw.W_i.copy(), "b_i_bw": bw.b_i.copy(),
            "W_c_bw": bw.W_c.copy(), "b_c_bw": bw.b_c.copy(),
            "W_o_bw": bw.W_o.copy(), "b_o_bw": bw.b_o.copy(),
            # Output weights
            "W_y": out_weights["W_y"].copy(),
            "b_y": out_weights["b_y"].copy(),
        }
