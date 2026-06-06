import numpy as np
from lstm_cell import LSTMCell
from utils import mse_loss, mse_loss_derivative

class LSTMNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.01, epochs = 50):
        # LSTM Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)

        # Output layer weights and biases
        self.W_y = np.zeros((output_size, hidden_size))
        self.b_y = np.ones(output_size)

        # Learning parameters
        self.learning_rate = learning_rate
        self.epochs = epochs

        # For backpropagation
        self.y_pred = []
        
    def train(self, X_train, y_train, X_val=None, y_val=None, patience = 10):
        total_loss = 0.0

        best_val_loss = float('inf')
        best_weights = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for X_seq, y_true in zip(X_train, y_train):
                # 1. Initialize cell and hidden states for the start of the sequence
                h = np.zeros(self.hidden_size)
                c = np.zeros(self.hidden_size)

                # 2. Forward pass through the sequence length, saving each step for BPTT
                states = []
                for t in range(len(X_seq)):
                    x_t = X_seq[t]
                    h_prev = h
                    c_prev = c
                    h, c = self.lstm_cell.forward_pass(x_t, h, c)
                    states.append({
                        'x_t': x_t,
                        'h_prev': h_prev,
                        'c_prev': c_prev,
                        'f_t': self.lstm_cell.f_t,
                        'i_t': self.lstm_cell.i_t,
                        'c_tilde': self.lstm_cell.c_tilde,
                        'c_t': self.lstm_cell.c_t,
                        'o_t': self.lstm_cell.o_t,
                    })

                # 3. Output layer calculation
                self.y_pred = self.W_y @ h + self.b_y

                # 4. Compute Loss and loss derivative (dy)
                loss = mse_loss(self.y_pred, y_true)
                epoch_loss += loss
                dy = mse_loss_derivative(self.y_pred, y_true)

                # 5. Backpropagation: Output layer gradients
                dh = self.W_y.T @ dy

                # Update output layer weights and biases using gradient descent
                self.W_y -= self.learning_rate * np.outer(dy, h)
                self.b_y -= self.learning_rate * dy

                # 6. LSTM cell backpropagation through time
                dc = np.zeros(self.hidden_size)
                for state in reversed(states):
                    dh, dc = self.lstm_cell.backward_pass(dh, dc, self.learning_rate, state=state)

            # Average the loss across all samples in the dataset
            epoch_loss /= len(X_train)
            total_loss = epoch_loss

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = 0.0
                for X_seq, y_true in zip(X_val, y_val):
                    h = np.zeros(self.hidden_size)
                    c = np.zeros(self.hidden_size)

                    for t in range(len(X_seq)):
                        h, c = self.lstm_cell.forward_pass(X_seq[t], h, c)

                    y_pred = self.W_y @ h + self.b_y
                    val_loss += mse_loss(y_pred, y_true)

                val_loss /= len(X_val)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save best weights
                    best_weights = {
                        'W_f': self.lstm_cell.W_f.copy(), 'b_f': self.lstm_cell.b_f.copy(),
                        'W_i': self.lstm_cell.W_i.copy(), 'b_i': self.lstm_cell.b_i.copy(),
                        'W_c': self.lstm_cell.W_c.copy(), 'b_c': self.lstm_cell.b_c.copy(),
                        'W_o': self.lstm_cell.W_o.copy(), 'b_o': self.lstm_cell.b_o.copy(),
                        'W_y': self.W_y.copy(),           'b_y': self.b_y.copy(),
                    }
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1} | Best Val Loss: {best_val_loss:.6f}")
                        # Restore best weights before returning
                        self._restore_weights(best_weights)
                        return total_loss

            if val_loss is not None:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {epoch_loss:.6f}")

        # If we never early stopped, still restore best weights seen during training
        if best_weights is not None:
            self._restore_weights(best_weights)

        return total_loss

    def _restore_weights(self, best_weights):
      self.lstm_cell.W_f = best_weights['W_f']
      self.lstm_cell.b_f = best_weights['b_f']
      self.lstm_cell.W_i = best_weights['W_i']
      self.lstm_cell.b_i = best_weights['b_i']
      self.lstm_cell.W_c = best_weights['W_c']
      self.lstm_cell.b_c = best_weights['b_c']
      self.lstm_cell.W_o = best_weights['W_o']
      self.lstm_cell.b_o = best_weights['b_o']
      self.W_y = best_weights['W_y']
      self.b_y = best_weights['b_y']

    def predict(self, X_seq, forecast_size=1):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
    
        for t in range(len(X_seq)):
            x_t = X_seq[t]
            h, c = self.lstm_cell.forward_pass(x_t, h, c)

        forecast = []
        last_known = X_seq[-1].copy()  

        for _ in range(forecast_size):
            h, c = self.lstm_cell.forward_pass(last_known, h, c)
            y_pred = self.W_y @ h + self.b_y
            forecast.append(y_pred[0])

            last_known[0] = y_pred[0]

        return np.array(forecast)


    def save_model(self, target):
        filename = f"{target}-lstm_model.npz"
        
        # We save all learnable parameters into a single .npz file
        np.savez(filename,
                 W_f=self.lstm_cell.W_f, b_f=self.lstm_cell.b_f,
                 W_i=self.lstm_cell.W_i, b_i=self.lstm_cell.b_i,
                 W_c=self.lstm_cell.W_c, b_c=self.lstm_cell.b_c,
                 W_o=self.lstm_cell.W_o, b_o=self.lstm_cell.b_o,
                 W_y=self.W_y, b_y=self.b_y)
        print(f"Model saved to {filename}")

    def load_model(self, target):
        filename = f"{target}-lstm_model.npz"

        data = np.load(filename)
        self.lstm_cell.W_f = data['W_f']
        self.lstm_cell.b_f = data['b_f']
        self.lstm_cell.W_i = data['W_i']
        self.lstm_cell.b_i = data['b_i']
        self.lstm_cell.W_c = data['W_c']
        self.lstm_cell.b_c = data['b_c']
        self.lstm_cell.W_o = data['W_o']
        self.lstm_cell.b_o = data['b_o']
        self.W_y = data['W_y']
        self.b_y = data['b_y']
        print(f"Model loaded from {filename}")