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

            states, h, y_pred = self._forward_sequence(X_seq)
            y_pred = y_pred.flatten()

            # Output layer forward from y_pred then compute yung loss (MSE)
            loss = huber_loss(y_pred, y_true, self.LOSS_DELTA)
            dy = huber_loss_derivative(y_pred, y_true, self.LOSS_DELTA).flatten()
            epoch_loss += loss

            # Backpropagation ng LSTM cell
            dh = self.network.output_layer.backward(dy, h)
            dc = np.zeros(self.network.hidden_size)

            T = len(X_seq)

            # Output layer backward (returns dh hidden state na ginagamit sa backpropagation)
            for t in reversed(range(T)):
                dh, dc = self.network.lstm_cell.backward_pass(dh, dc, state=states[t])
            
            self._clip_gradient()
            self.network.lstm_cell.update_weights(self.learning_rate)
            self.network.output_layer.update_weights(self.learning_rate)
        
        return epoch_loss / len(X_train)


    def _clip_gradient(self, clip_threshold = 5.0):
        # Gradient clipping (prevents grdient explosion)
        cell = self.network.lstm_cell
        all_grads = [
            cell.dW_f, cell.dW_i, cell.dW_c, cell.dW_o, cell.db_f, cell.db_i, cell.db_c, cell.db_o,
            self.network.output_layer.dW_y, self.network.output_layer.db_y,
        ]
        for grad in all_grads:
            np.clip(grad, -clip_threshold, clip_threshold, out=grad)


    def _forward_sequence(self, X_seq):
        cell = self.network.lstm_cell

        T = len(X_seq)

        states = []
        h = np.zeros(self.network.hidden_size)
        c = np.zeros(self.network.hidden_size)

        # Forward pass through input sequence 
        for t in range(T):
            h, c, state = cell.forward_pass(X_seq[t], h, c)
            if self.dropout_rate:
                mask = (RNG.random(h.shape) > self.dropout_rate).astype(float)
                h = h * mask / (1 - self.dropout_rate)
            
            states.append(state)

        y_pred = self.network.output_layer.forward(h)

        return states, h, y_pred

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

            for (X_seq, y_true) in zip(X, y):
                y_true = np.array(y_true)
                y_pred = np.array(self.network.predict(X_seq))


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
