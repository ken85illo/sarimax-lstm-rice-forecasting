import numpy as np
from lstm import LSTMNetwork
from .checkpoint import ModelCheckpoint

class Trainer:
    def __init__(self, network: LSTMNetwork, learning_rate: float = 0.001, epochs: int = 300, patience: int = 20):
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience

    # == Training Function for LSTM ==
    def train(self, X_train, y_train, X_val=None, y_val=None) -> float:
        checkpoint = ModelCheckpoint()
        epochs_no_improve = 0
        total_loss = 0.0

        for epoch in range(self.epochs):
            epoch_loss = self._train_one_epoch(X_train, y_train)
            total_loss = epoch_loss

            val_loss = self._evaluate(X_val, y_val) if X_val is not None else None

            self._log(epoch, epoch_loss, val_loss)

            if val_loss is not None:
                improved = checkpoint.update(self.network, val_loss) # Check if may improvement

                # Early stop function (nakabase sa patience)
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(
                            f"Early stopping at epoch {epoch + 1} "
                            f"| Best Val Loss: {checkpoint.best_loss:.6f}"
                        )
                        checkpoint.restore(self.network)
                        return total_loss

        # Restore yung best weights sa validation
        if X_val is not None:
            checkpoint.restore(self.network)
        return total_loss


    # == Training function for one epoch ==
    def _train_one_epoch(self, X_train, y_train) -> float:
        epoch_loss = 0.0

        for X_seq, y_true in zip(X_train, y_train):
            h, c, states = self._forward_sequence(X_seq)

            # Output layer forward then compute yung loss (MSE)
            y_pred = self.network.output_layer.forward(h)
            loss = self.network.output_layer.loss(y_pred, y_true)
            dy = self.network.output_layer.loss_derivative(y_pred, y_true)
            epoch_loss += loss

            # Output layer backward (returns dh hidden state na ginagamit sa backpropagation)
            dh = self.network.output_layer.backward(dy, h, self.learning_rate)

            # Backpropagation ng LSTM cell
            dc = np.zeros(self.network.hidden_size)
            for state in reversed(states):
                dh, dc = self.network.lstm_cell.backward_pass(
                    dh, dc, self.learning_rate, state=state
                )

            self.network.lstm_cell.update_weights(self.learning_rate)
            

        return epoch_loss / len(X_train)

    def _forward_sequence(self, X_seq):
        h = np.zeros(self.network.hidden_size)
        c = np.zeros(self.network.hidden_size)
        cell = self.network.lstm_cell
        states = []

        for t in range(len(X_seq)):
            x_t = X_seq[t]
            h_prev, c_prev = h, c
            h, c = cell.forward_pass(x_t, h, c)
            states.append({
                "x_t":     x_t,
                "h_prev":  h_prev,
                "c_prev":  c_prev,
                "f_t":     cell.f_t,
                "i_t":     cell.i_t,
                "c_tilde": cell.c_tilde,
                "c_t":     cell.c_t,
                "o_t":     cell.o_t,
            })

        return h, c, states

    def _evaluate(self, X_val, y_val) -> float:
        # Compute average MSE on the validation set (no weight updates)
        total = 0.0
        for X_seq, y_true in zip(X_val, y_val):
            h = np.zeros(self.network.hidden_size)
            c = np.zeros(self.network.hidden_size)
            for t in range(len(X_seq)):
                h, c = self.network.lstm_cell.forward_pass(X_seq[t], h, c)
            y_pred = self.network.output_layer.forward(h)
            total += self.network.output_layer.loss(y_pred, y_true)
        return total / len(X_val)

    # == Log function for train and validation loss per epoch ==
    @staticmethod
    def _log(epoch: int, train_loss: float, val_loss: float | None):
        if val_loss is not None:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f}")