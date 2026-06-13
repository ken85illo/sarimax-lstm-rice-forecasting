import numpy as np
from lstm import LSTMNetwork
import matplotlib.pyplot as plt

from utils import huber_loss, huber_loss_derivative, RNG
from .checkpoint import ModelCheckpoint


class Trainer:
    LOSS_DELTA = 1.0

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
            y_true = y_true.flatten()

            fw_states, bw_states, h_concat_final, y_pred = self._forward_sequence(X_seq, len(y_true))
            y_pred = y_pred.flatten()

            # Output layer forward from y_pred then compute yung loss (MSE)
            loss = huber_loss(y_pred, y_true, self.LOSS_DELTA)
            dy = huber_loss_derivative(y_pred, y_true, self.LOSS_DELTA).flatten()
            epoch_loss += loss

            dh_concat = self.network.output_layer.backward(dy, h_concat_final)

            # Backpropagation ng LSTM cell
            dh_fw = dh_concat[:self.network.hidden_size].copy()
            dh_bw = dh_concat[self.network.hidden_size:].copy()
            
            dc_fw = np.zeros(self.network.hidden_size)
            dc_bw = np.zeros(self.network.hidden_size)

            T = len(X_seq)

            # Output layer backward (returns dh hidden state na ginagamit sa backpropagation)
            for t in reversed(range(T)):
                dh_fw, dc_fw = self.network.forward_cell.backward_pass(dh_fw, dc_fw, state=fw_states[t])

            for t in range(T):
                dh_bw, dc_bw = self.network.backward_cell.backward_pass(dh_bw, dc_bw, state=bw_states[t])
            
            self._clip_gradient()
            self.network.forward_cell.update_weights(self.learning_rate)
            self.network.backward_cell.update_weights(self.learning_rate)
            self.network.output_layer.update_weights(self.learning_rate)
        
        return epoch_loss / len(X_train)


    def _clip_gradient(self, clip_threshold = 5.0):
        # Gradient clipping (prevents grdient explosion)
        fw, bw = self.network.forward_cell, self.network.backward_cell
        all_grads = [
            fw.dW_f, fw.dW_i, fw.dW_c, fw.dW_o, fw.db_f, fw.db_i, fw.db_c, fw.db_o,
            bw.dW_f, bw.dW_i, bw.dW_c, bw.dW_o, bw.db_f, bw.db_i, bw.db_c, bw.db_o,
            self.network.output_layer.dW_y, self.network.output_layer.db_y,
        ]
        for grad in all_grads:
            np.clip(grad, -clip_threshold, clip_threshold, out=grad)


    def _forward_sequence(self, X_seq, steps):
        fw_cell = self.network.forward_cell
        bw_cell = self.network.backward_cell

        T = len(X_seq)

        fw_states = []
        bw_states = [None] * T
        
        h_fw_list = []
        h_bw_list = [None] * T

        h_fw = np.zeros(self.network.hidden_size)
        c_fw = np.zeros(self.network.hidden_size)

        # Forward pass through input sequence 
        for t in range(T):
            h_fw, c_fw, state = fw_cell.forward_pass(X_seq[t], h_fw, c_fw)
            if self.dropout_rate:
                mask = (RNG.random(h_fw.shape) > self.dropout_rate).astype(float)
                h_fw = h_fw * mask / (1 - self.dropout_rate)
            
            fw_states.append(state)
            h_fw_list.append(h_fw.copy())

        h_bw = np.zeros(self.network.hidden_size)
        c_bw = np.zeros(self.network.hidden_size)

        # Backward pass (after to nung forward pass sa input sequence)
        for t in reversed(range(T)):
            h_bw, c_bw, state = bw_cell.forward_pass(X_seq[t], h_bw, c_bw)
            if self.dropout_rate:
                mask = (RNG.random(h_bw.shape) > self.dropout_rate).astype(float)
                h_bw = h_bw * mask / (1 - self.dropout_rate)
            
            bw_states[t] = state
            h_bw_list[t] = h_bw.copy()

        h_concat_final = np.concatenate((h_fw_list[-1], h_bw_list[0]), axis=0)
        y_pred = self.network.output_layer.forward(h_concat_final)

        return fw_states, bw_states, h_concat_final, y_pred

    def _evaluate(self, X_val, y_val) -> float:
        # Compute average MSE on the validation set (no weight updates)
        total = 0.0
        for X_seq, y_true in zip(X_val, y_val):
            y_pred = self.network.predict(X_seq)
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()

            total += huber_loss(y_pred, y_true, self.LOSS_DELTA)
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

                y_pred = self.network.predict(X_seq)
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
