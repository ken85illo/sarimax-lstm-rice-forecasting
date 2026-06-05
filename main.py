import numpy as np
import pandas as pd

# Takes x and returns value between 0 and 1
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Takes x and returns value between -1 and 1
def tanh_function(x):
    return np.tanh(x)

# Derivative of tanh function
def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2

# Derivative of sigmoid function
def sigmoid_derivative(x):
    s = sigmoid_function(x)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_pred.shape[0]


def create_sequences(data, lookback):
    data = np.array(data)  # ← converts Series or list to numpy array

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

class LSTMCell:
    RNG_MEAN = 0.0
    RNG_STD_DEV = 0.1

    def __init__(self, input_size, hidden_size):
        self.rng = np.random.default_rng()

        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = 0.1

        # Forget gate learnables
        self.W_f = scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, (hidden_size, hidden_size + input_size))
        self.b_f = scale * np.ones(hidden_size)

        # Input gate learnables
        self.W_i= scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, (hidden_size, hidden_size + input_size))
        self.b_i = scale * np.ones(hidden_size)

        # C tilde learnables
        self.W_c = scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, (hidden_size, hidden_size + input_size))
        self.b_c = scale * np.ones(hidden_size)

        # Output gate learnables
        self.W_o = scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, (hidden_size, hidden_size + input_size))
        self.b_o = scale * np.ones(hidden_size)


    def forward_pass(self, x_t, h_prev, c_prev):
        # Storage for backpropagation
        self.x_t = x_t
        self.h_prev = h_prev
        self.c_prev = c_prev

        # print(f"h_prev: {h_prev}")
        # print(f"X_t: {self.x_t}")
        # Concatenate h_prev (hidden state) and x_t (input at current time step)
        X_t = np.concatenate((self.h_prev, self.x_t), axis=0)

        # Forget gate calculation
        self.f_t = sigmoid_function((self.W_f @ X_t) + self.b_f)

        # Input gate calculation
        self.i_t = sigmoid_function((self.W_i @ X_t) + self.b_i)

        # Cell state candidate calculation
        self.c_tilde = tanh_function((self.W_c @ X_t) + self.b_c)

        # New cell state calculation
        self.c_t = self.i_t * self.c_tilde + self.f_t * self.c_prev

        # Output gate calculation
        self.o_t = sigmoid_function((self.W_o @ X_t) + self.b_o)

        # New hidden state calculation
        self.h_t = self.o_t * tanh_function(self.c_t)

        return self.h_t, self.c_t

    def backward_pass(self, dh_next, dc_next, learning_rate, state=None, clip_value = 5.0):
        if state is not None:
            self.x_t = state['x_t']
            self.h_prev = state['h_prev']
            self.c_prev = state['c_prev']
            self.f_t = state['f_t']
            self.i_t = state['i_t']
            self.c_tilde = state['c_tilde']
            self.c_t = state['c_t']
            self.o_t = state['o_t']

        # Gradient of the output gate
        do_t  = dh_next * tanh_function(self.c_t) * sigmoid_derivative(self.o_t)

        # Gradient of the cell state
        dc_t = dh_next * self.o_t * tanh_derivative(self.c_t) + dc_next

        # Gradient of the input gate
        di_t = dc_t * self.c_tilde * sigmoid_derivative(self.i_t)

        # Gradient of the cell state candidate
        dc_tilde = dc_t * self.i_t * tanh_derivative(self.c_tilde)

        # Gradient of the forget gate
        df_t = dc_t * self.c_prev * sigmoid_derivative(self.f_t)

        # Concatenate h_prev (hidden state) and x_t (input at current time step)
        X_t = np.concatenate((self.h_prev, self.x_t), axis=0)

        # Gradient clipping to prevent exploding gradients
        do_t = np.clip(do_t, -clip_value, clip_value)
        di_t = np.clip(di_t, -clip_value, clip_value)
        dc_tilde = np.clip(dc_tilde, -clip_value, clip_value)
        df_t = np.clip(df_t, -clip_value, clip_value)


        # Update weights and biases for forget gate
        self.W_f -= learning_rate * np.outer(df_t,X_t) # Transposed X_t
        self.b_f -= learning_rate * df_t

        # Update weights and biases for input gate
        self.W_i -= learning_rate * np.outer(di_t, X_t) # Transposed X_t
        self.b_i -= learning_rate * di_t

        # Update weights and biases for cell state candidate
        self.W_c -= learning_rate * np.outer(dc_tilde, X_t) # Transposed X_t
        self.b_c -= learning_rate * dc_tilde

        # Update weights and biases for output gate
        self.W_o -= learning_rate * np.outer(do_t, X_t) # Transposed X_t
        self.b_o -= learning_rate * do_t

        # Compute gradients with respect to inputs for backpropagation to earlier layers
        # Optimized dh_prev calculation using vectorized operations
        dh_prev = self.W_f[:, :self.hidden_size].T @ df_t + \
                  self.W_i[:, :self.hidden_size].T @ di_t + \
                  self.W_c[:, :self.hidden_size].T @ dc_tilde + \
                  self.W_o[:, :self.hidden_size].T @ do_t

        dc_prev = dc_t * self.f_t

        return dh_prev, dc_prev


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

            # ── Logging ─────────────────────────────
            # if (epoch + 1) % 10 == 0 or epoch == 0:
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

        output = self.W_y @ h + self.b_y
            
        return output


    def save_model(self, filename="lstm_model.npz"):
        # We save all learnable parameters into a single .npz file
        np.savez(filename,
                 W_f=self.lstm_cell.W_f, b_f=self.lstm_cell.b_f,
                 W_i=self.lstm_cell.W_i, b_i=self.lstm_cell.b_i,
                 W_c=self.lstm_cell.W_c, b_c=self.lstm_cell.b_c,
                 W_o=self.lstm_cell.W_o, b_o=self.lstm_cell.b_o,
                 W_y=self.W_y, b_y=self.b_y)
        print(f"Model saved to {filename}")

    def load_model(self, filename="lstm_model.npz"):
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

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        # Calculate min and max per feature
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        # Apply the formula: (x - min) / (max - min)
        # We add a tiny epsilon to avoid division by zero
        return (data - self.min) / (self.max - self.min + 1e-8)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        # Useful for converting predictions back to original price scale
        return scaled_data * (self.max - self.min + 1e-8) + self.min

def train_on_well_milled_high():
    df_raw_rice = pd.read_csv("well_milled_rice_daily_preprocessed.csv")
    df_raw_rice['Date'] = pd.to_datetime(df_raw_rice['Date'])
    df_raw_rice = df_raw_rice.sort_values('Date').reset_index(drop=True)

    # 70-20-10 split
    train_split_count = 0.7
    validation_split_count = 0.2

    # Row split calculation for training - validation - testing
    total_rows = len(df_raw_rice)

    train_size = int(total_rows * train_split_count)
    val_size = int(total_rows * validation_split_count)

    # Split rice price data
    train_val_rice_df = df_raw_rice.iloc[:train_size + val_size].copy()
    train_rice_df = df_raw_rice.iloc[:train_size].copy()
    val_rice_df = df_raw_rice.iloc[train_size : train_size + val_size].copy()

    high_train_val_df = train_val_rice_df["Well-Milled High"]
    high_train_df = train_rice_df["Well-Milled High"]
    high_val_df = val_rice_df["Well-Milled High"]

    # Fitting Min-Max Scaler
    scaler = MinMaxScaler()
    scaler.fit(high_train_val_df.values.reshape(-1, 1))

    scaled_df_train = scaler.transform(high_train_df.values.reshape(-1, 1)) # Fit on training data
    scaled_df_val = scaler.transform(high_val_df.values.reshape(-1, 1))     # Transform validation data using the fitted scaler

    # Create sequences (adjust lookback as needed)
    lookback = 14
    X_train, y_train = create_sequences(scaled_df_train, lookback)
    X_val, y_val = create_sequences(scaled_df_val, lookback)

    # TODO: Instantiate and train your network
    network = LSTMNetwork(input_size=1, hidden_size=64, output_size=1, learning_rate=0.1, epochs=200)
    network.train( X_train, y_train, X_val, y_val, patience=20)
    network.save_model()

    

def train_on_residual_high():
    # TODO: Specify your CSV file paths here
    # train_csv = pd.read_csv("sarimax_train_residuals_Well-Milled_High.csv")
    # val_csv = pd.read_csv("sarimax_val_residuals_Well-Milled_High.csv")

    train_csv = pd.read_csv("sarimax_train_residuals_Well-Milled_High.csv")
    val_csv = pd.read_csv("sarimax_val_residuals_Well-Milled_High.csv")

    df_train = train_csv["Well-Milled High_train_residual"]
    df_val = val_csv["Well-Milled High_val_residual"]

    scaler = MinMaxScaler()
    scaled_df_train = scaler.fit_transform(df_train.values.reshape(-1, 1)) # Fit on training data
    scaled_df_val = scaler.transform(df_val.values.reshape(-1, 1))     # Transform validation data using the fitted scaler

    # Create sequences (adjust lookback as needed)
    lookback = 5
    X_train, y_train = create_sequences(scaled_df_train, lookback)
    X_val, y_val = create_sequences(scaled_df_val, lookback)

    # TODO: Instantiate and train your network
    network = LSTMNetwork(input_size=1, hidden_size=64, output_size=1, learning_rate=0.1, epochs=200)
    network.train( X_train, y_train, X_val, y_val, patience=20)
    network.save_model()

def test_overfitting():
    # 1. Setup: 3 inputs, 2 hidden neurons, 1 output
    network = LSTMNetwork(input_size=3, hidden_size=64, output_size=1, learning_rate=0.1, epochs=200)

    # 2. Dummy data: A single sequence of 5 time steps
    X_sample = np.random.randn(5, 3) # 5 steps, 3 features
    y_sample = np.array([0.8])       # Target

    # Wrap in list so it matches the expected iterable structure
    X_train = [X_sample]
    y_train = [y_sample]

    X_val = [X_sample]
    y_val = [y_sample]

    print(X_train)
    print(y_train)

    #3. Train
    print("Starting sanity check (loss should decrease)...")
    network.train( X_train, y_train, X_val, y_val, patience=20)
    print("\n\n===== FINISH TRAINING =====\n\n")

    #4. Predict
    final_pred = network.predict(X_sample)
    print(f"Target: {y_sample}, Prediction: {final_pred}")

def test_prediction():
    df_raw_rice = pd.read_csv("well_milled_rice_daily_preprocessed.csv")
    df_raw_rice['Date'] = pd.to_datetime(df_raw_rice['Date'])
    df_raw_rice = df_raw_rice.sort_values('Date').reset_index(drop=True)

    # 70-20-10 split
    train_split_count = 0.7
    validation_split_count = 0.2

    # Row split calculation for training - validation - testing
    total_rows = len(df_raw_rice)

    train_size = int(total_rows * train_split_count)
    val_size = int(total_rows * validation_split_count)

    # Split rice price data
    scaler = MinMaxScaler()
    train_val_rice_df = df_raw_rice.iloc[:train_size + val_size].copy()
    high_train_val_df = train_val_rice_df["Well-Milled High"]
    scaler.fit(high_train_val_df.values.reshape(-1, 1)) # Fit on training data

    # Test Dataset and min-max transform
    test_rice_df = df_raw_rice.iloc[train_size + val_size:].copy()
    high_test_df = test_rice_df["Well-Milled High"]
    scaled_df_test = scaler.transform(high_test_df.values.reshape(-1, 1)) # Fit on training data

    # Parameters and sequence creation
    lookback = 4
    X_test, y_test = create_sequences(scaled_df_test, lookback)
    test_dates = test_rice_df['Date'].iloc[lookback:].reset_index(drop=True)

    network = LSTMNetwork(input_size=1, hidden_size=64, output_size=1, learning_rate=0.1, epochs=200)
    network.load_model()

    for i, (X_seq, y_actual) in enumerate(zip(X_test, y_test)):
        date = test_dates.iloc[i]
        input_seq = X_seq
        input_inverse_seq = scaler.inverse_transform(input_seq)
        actual = scaler.inverse_transform(y_actual)
        predicted = scaler.inverse_transform(network.predict(input_seq))
        print(f"Date: {date.date()}\nInput: {input_inverse_seq}\nActual: {actual}\nPredicted: {predicted}\n")

        if i == 5:
            break

    
    


    

if __name__ == "__main__":
    print(f"SANITY CHECK ON SYNTHETIC DATA ")
    test_overfitting()
    
    test_prediction()
    
    # train_on_well_milled_high()

    # print(f"\nACTUAL")
    # train_on_residual_high()
    

