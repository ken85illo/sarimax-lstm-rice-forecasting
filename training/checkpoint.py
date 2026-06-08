
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
        
        cell_1 = network.lstm_cell_1
        cell_2 = network.lstm_cell_2

        # Mainly used to restore yung best weights if no improvement na yung future epochs

        # Layer 1
        cell_1.W_f, cell_1.b_f = self._weights['c1_W_f'], self._weights['c1_b_f']
        cell_1.W_i, cell_1.b_i = self._weights['c1_W_i'], self._weights['c1_b_i']
        cell_1.W_c, cell_1.b_c = self._weights['c1_W_c'], self._weights['c1_b_c']
        cell_1.W_o, cell_1.b_o = self._weights['c1_W_o'], self._weights['c1_b_o']
        
        # Layer 2
        cell_2.W_f, cell_2.b_f = self._weights['c2_W_f'], self._weights['c2_b_f']
        cell_2.W_i, cell_2.b_i = self._weights['c2_W_i'], self._weights['c2_b_i']
        cell_2.W_c, cell_2.b_c = self._weights['c2_W_c'], self._weights['c2_b_c']
        cell_2.W_o, cell_2.b_o = self._weights['c2_W_o'], self._weights['c2_b_o']

        network.output_layer.set_weights(
            {"W_y": self._weights["W_y"], "b_y": self._weights["b_y"]}
        )

    # == Returns the current weights and biases ng cell == 
    @staticmethod
    def _snapshot(network):
        cell_1 = network.lstm_cell_1
        cell_2 = network.lstm_cell_2

        out_weights = network.output_layer.get_weights()

        return {
            'c1_W_f': cell_1.W_f.copy(), 'c1_b_f': cell_1.b_f.copy(),
            'c1_W_i': cell_1.W_i.copy(), 'c1_b_i': cell_1.b_i.copy(),
            'c1_W_c': cell_1.W_c.copy(), 'c1_b_c': cell_1.b_c.copy(),
            'c1_W_o': cell_1.W_o.copy(), 'c1_b_o': cell_1.b_o.copy(),
            
            'c2_W_f': cell_2.W_f.copy(), 'c2_b_f': cell_2.b_f.copy(),
            'c2_W_i': cell_2.W_i.copy(), 'c2_b_i': cell_2.b_i.copy(),
            'c2_W_c': cell_2.W_c.copy(), 'c2_b_c': cell_2.b_c.copy(),
            'c2_W_o': cell_2.W_o.copy(), 'c2_b_o': cell_2.b_o.copy(),
            "W_y": out_weights["W_y"].copy(),
            "b_y": out_weights["b_y"].copy(),
        }