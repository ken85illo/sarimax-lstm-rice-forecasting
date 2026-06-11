import numpy as np
from lstm import LSTMNetwork
import matplotlib.pyplot as plt

from utils import mse_loss, mse_loss_derivative, RNG
from .checkpoint import ModelCheckpoint

class Trainer:
    def __init__(self, network: LSTMNetwork, learning_rate: float = 0.001, epochs: int = 300, patience: int = 20, dropout_rate = 0.1):
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.dropout_rate = dropout_rate

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
            states, step_hs, y_pred = self._forward_sequence(X_seq, len(y_true))
            y_pred = y_pred.flatten()

            # Output layer forward from y_pred then compute yung loss (MSE)
            loss = mse_loss(y_pred, y_true)
            dy = mse_loss_derivative(y_pred, y_true).flatten()
            epoch_loss += loss

            # Backpropagation ng LSTM cell
            dh = np.zeros(self.network.hidden_size)
            dc = np.zeros(self.network.hidden_size)

            encoder_states = states[:len(X_seq)]   
            decoder_states = states[len(X_seq):]   

            # Output layer backward (returns dh hidden state na ginagamit sa backpropagation)
            for step in reversed(range(len(y_true))):
                dy_step = np.atleast_1d(dy[step])
                dh += self.network.output_layer.backward(dy_step, step_hs[step])
                dh, dc = self.network.lstm_cell.backward_pass(
                    dh, dc, self.learning_rate, state=decoder_states[step]
                )

            # Then backprop through encoder
            for state in reversed(encoder_states):
                dh, dc = self.network.lstm_cell.backward_pass(
                    dh, dc, self.learning_rate, state=state
                )
            
            self._clip_gradient()
            self.network.lstm_cell.update_weights(self.learning_rate)
            self.network.output_layer.update_weights(self.learning_rate)
        
        return epoch_loss / len(X_train)


    def _clip_gradient(self, clip_threshold = 5.0):
        # Gradient clipping (prevents grdient explosion)
        all_grads = [
            self.network.lstm_cell.dW_f, self.network.lstm_cell.dW_i,
            self.network.lstm_cell.dW_c, self.network.lstm_cell.dW_o,
            self.network.lstm_cell.db_f, self.network.lstm_cell.db_i,
            self.network.lstm_cell.db_c, self.network.lstm_cell.db_o,
            self.network.output_layer.dW_y, self.network.output_layer.db_y,
        ]
        total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
        if total_norm > clip_threshold:
            scale = clip_threshold / (total_norm + 1e-8)
            for g in all_grads:
                g[:] *= scale


    def _forward_sequence(self, X_seq, steps):
        cell = self.network.lstm_cell
        states = []
        step_hs = []

        h = np.zeros(self.network.hidden_size)
        c = np.zeros(self.network.hidden_size)

        # Forward pass through input sequence 
        # (kunin yung last na hidden and cell state)
        sequence = list(X_seq.copy())  
        for t in range(len(sequence)):
            x_t = sequence[t]

            h_prev, c_prev = h.copy(), c.copy()
            h, c = cell.forward_pass(x_t, h, c)

            # Dropout Rate
            if self.dropout_rate is not None:
                mask = (RNG.random(h.shape) > self.dropout_rate).astype(float)
                h = h * mask / (1 - self.dropout_rate)

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

        # Get predictions (after to nung forward pass sa input sequence)
        predictions = []
        for _ in range(steps):
            step_hs.append(h.copy())
            pred = self.network.output_layer.forward(h)
            predictions.append(pred)

            x_t = sequence[-1].copy()
            x_t[0] = pred[0]
            
            sequence.append(x_t)
            sequence.pop(0)  

            h_prev, c_prev = h.copy(), c.copy()
            h, c = cell.forward_pass(x_t, h, c)

            # Dropout Rate
            if self.dropout_rate is not None:
                mask = (RNG.random(h.shape) > self.dropout_rate).astype(float)
                h = h * mask / (1 - self.dropout_rate)

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

        return states, step_hs, np.array(predictions)

    def _evaluate(self, X_val, y_val) -> float:
        # Compute average MSE on the validation set (no weight updates)
        total = 0.0
        for X_seq, y_true in zip(X_val, y_val):
            y_pred = self.network.predict(X_seq, len(y_true))
            total += mse_loss(y_pred, y_true)
        return total / len(X_val)


    # == Plot fitted train predictions and val predictions vs actuals ==
    def plot_predictions(
        self,
        target,
        title,
        scaler,
        X_train, y_train,
        X_val=None, y_val=None,
    ):
        def run_predictions(X, y):
            preds, actuals = [], []

            for i, (X_seq, y_true) in enumerate(zip(X, y)):
                if i % len(y_true) != 0:   # Every 14 days
                    continue

                y_pred = self.network.predict(X_seq, len(y_true))
                preds.extend(y_pred.flatten())
                actuals.extend(y_true.flatten())
 
            preds = np.array(preds).reshape(-1, 1)
            actuals = np.array(actuals).reshape(-1, 1)
 
            preds = scaler.inverse_transform_feature(preds, 0).flatten()
            actuals = scaler.inverse_transform_feature(actuals, 0).flatten()
 
            return preds, actuals
 
        train_preds, train_actuals = run_predictions(X_train, y_train)
 
        fig, axes = plt.subplots(
            1 if X_val is None else 2,
            1,
            figsize=(14, 5 if X_val is None else 10),
            sharex=False,
        )
 
        # Make axes always iterable
        if X_val is None:
            axes = [axes]
 
        # Train plot 
        ax = axes[0]
        ax.plot(train_actuals, label="Actual", color="#2563eb", linewidth=1.5)
        ax.plot(train_preds,   label="Fitted (Train)", color="#ff0000", linewidth=1.5)
        ax.set_title(f"{title} (Training Set)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
 
        # Validation plot 
        if X_val is not None:
            val_preds, val_actuals = run_predictions(X_val, y_val)
            ax = axes[1]
            ax.plot(val_actuals, label="Actual",             color="#2563eb", linewidth=1.5)
            ax.plot(val_preds,   label="Predicted (Val)",    color="#ff0000", linewidth=1.5)
            ax.set_title(f"{title} (Validation Set)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig(f"output/{target}_training_val_plot.png")


    # == Log function for train and validation loss per epoch ==
    @staticmethod
    def _log(epoch: int, train_loss: float, val_loss: float | None):
        if val_loss is not None:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.6f}")
