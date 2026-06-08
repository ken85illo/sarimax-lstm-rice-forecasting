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
        checkpoint.restore(self.network)
        return total_loss


    # == Training function for one epoch ==
    def _train_one_epoch(self, X_train, y_train) -> float:
        epoch_loss = 0.0

        for X_seq, y_true in zip(X_train, y_train):
            h2, c2, states = self._forward_sequence(X_seq)

            # Output layer forward then compute yung loss (MSE)
            y_pred = self.network.output_layer.forward(h2)
            loss = self.network.output_layer.loss(y_pred, y_true)
            dy = self.network.output_layer.loss_derivative(y_pred, y_true)
            epoch_loss += loss

            # Output layer backward (returns dh hidden state na ginagamit sa backpropagation)
            dh2 = self.network.output_layer.backward(dy, h2, self.learning_rate)

            # Backpropagation ng LSTM cell
            dc2 = np.zeros(self.network.hidden_size)
            dh1 = np.zeros(self.network.hidden_size)
            dc1 = np.zeros(self.network.hidden_size)

            for state_1, state_2 in reversed(states):
                dx2, dh2, dc2 = self.network.lstm_cell_2.backward_pass(
                    dh2, dc2, self.learning_rate, state=state_2
                )

                dh1 += dx2

                dx1, dh1, dc1 = self.network.lstm_cell_1.backward_pass(
                    dh1, dc1, self.learning_rate, state=state_1
                )

        return epoch_loss / len(X_train)

    def _forward_sequence(self, X_seq):
        h1, c1 = np.zeros(self.network.hidden_size), np.zeros(self.network.hidden_size)
        h2, c2 = np.zeros(self.network.hidden_size), np.zeros(self.network.hidden_size)

        cell_1 = self.network.lstm_cell_1
        cell_2 = self.network.lstm_cell_2

        states = []

        for t in range(len(X_seq)):
            x_t = X_seq[t]

            h1_prev, c1_prev = h1, c1
            h2_prev, c2_prev = h2, c2

            h1, c1 = cell_1.forward_pass(x_t, h1, c1)
            h2, c2 = cell_2.forward_pass(h1, h2, c2)

            state_1 = {
                'x_t': x_t, 'h_prev': h1_prev, 'c_prev': c1_prev,
                'f_t': cell_1.f_t, 'i_t': cell_1.i_t,
                'c_tilde': cell_1.c_tilde, 'c_t': cell_1.c_t, 'o_t': cell_1.o_t
            }
                    
            state_2 = {
                'x_t': h1,  # Critical: Layer 2's input was h1
                'h_prev': h2_prev, 'c_prev': c2_prev,
                'f_t': cell_2.f_t, 'i_t': cell_2.i_t,
                'c_tilde': cell_2.c_tilde, 'c_t': cell_2.c_t, 'o_t': cell_2.o_t
            }

            states.append((state_1, state_2))

        return h2, c2, states

    def _evaluate(self, X_val, y_val) -> float:
        # Compute average MSE on the validation set (no weight updates)
        total = 0.0
        cell_1 = self.network.lstm_cell_1
        cell_2 = self.network.lstm_cell_2

        for X_seq, y_true in zip(X_val, y_val):
            h1, c1 = np.zeros(self.network.hidden_size), np.zeros(self.network.hidden_size)
            h2, c2 = np.zeros(self.network.hidden_size), np.zeros(self.network.hidden_size)

            for t in range(len(X_seq)):
                h1, c1 = cell_1.forward_pass(X_seq[t], h1, c1)
                h2, c2 = cell_2.forward_pass(h1, h2, c2)

            y_pred = self.network.output_layer.forward(h2)
            total += self.network.output_layer.loss(y_pred, y_true)
        return total / len(X_val)

    # == Log function for train and validation loss per epoch ==
    @staticmethod
    def _log(epoch: int, train_loss: float, val_loss: float | None):
        if val_loss is not None:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f}")