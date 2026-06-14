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
        cell = network.lstm_cell
        cell.W_f, cell.b_f = self._weights["W_f"].copy(), self._weights["b_f"].copy()
        cell.W_i, cell.b_i = self._weights["W_i"].copy(), self._weights["b_i"].copy()
        cell.W_c, cell.b_c = self._weights["W_c"].copy(), self._weights["b_c"].copy()
        cell.W_o, cell.b_o = self._weights["W_o"].copy(), self._weights["b_o"].copy()

        network.output_layer.set_weights(
            {"W_y": self._weights["W_y"].copy(), "b_y": self._weights["b_y"].copy()}
        )

    # == Returns the current weights and biases ng cell == 
    @staticmethod
    def _snapshot(network):
        cell = network.lstm_cell
        out_weights = network.output_layer.get_weights()

        return {
            # Forward weights
            "W_f": cell.W_f.copy(), "b_f": cell.b_f.copy(),
            "W_i": cell.W_i.copy(), "b_i": cell.b_i.copy(),
            "W_c": cell.W_c.copy(), "b_c": cell.b_c.copy(),
            "W_o": cell.W_o.copy(), "b_o": cell.b_o.copy(),
            # Output weights
            "W_y": out_weights["W_y"].copy(),
            "b_y": out_weights["b_y"].copy(),
        }
