import numpy as np
import math

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
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i]) # (5 x 6)
        y.append(data[i, 0]) # (1 x 1)
    return np.array(X), np.array(y)        


# class LSTM():
#     def __init__(self, n_neurons):
#         self.n_neurons = n_neurons
#
#         # Forget gate learnables
#         self.Uf = 0.1 * np.random.randn(n_neurons, 1)
#         self.bf = 0.1 * np.random.randn(n_neurons, 1)
#         self.Wf = 0.1 * np.random.randn(n_neurons, n_neurons)
#
#         # Input gate learnables
#         self.Ui = 0.1 * np.random.randn(n_neurons, 1)
#         self.bi = 0.1 * np.random.randn(n_neurons, 1)
#         self.Wi = 0.1 * np.random.randn(n_neurons, n_neurons)
#
#         # Output gate learnables
#         self.Uo = 0.1 * np.random.randn(n_neurons, 1)
#         self.bo = 0.1 * np.random.randn(n_neurons, 1)
#         self.Wo = 0.1 * np.random.randn(n_neurons, n_neurons)
#
#         # C tilde learnables
#         self.Ug = 0.1 * np.random.randn(n_neurons, 1)
#         self.bg = 0.1 * np.random.randn(n_neurons, 1)
#         self.Wg = 0.1 * np.random.randn(n_neurons, n_neurons)

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

        # self.W_y = scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, (input_size, hidden_size))
        # self.b_y = scale * np.ones((hidden_size, 1))


    def forward_pass(self, x_t, h_prev, c_prev):
        # Storage for backpropagation
        self.x_t = x_t
        self.h_prev = h_prev
        self.c_prev = c_prev

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
    
    def backward_pass(self, dh_next, dc_next, learning_rate, clip_value = 5.0):
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


        # print(f"c_prev: {self.c_prev}")
        # print(f"dc_t: {dc_t}")
        # print(f"df_t: {df_t}")
        # print(f"X_t: {X_t}")
        # print(f"X_t.T: {X_t.T}")
        # Di pala gumagana yung Transpose sa 1D array kaya daw palitan ng np.outer

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
        dX_t  = np.zeros((self.input_size, 1))
        dh_prev = np.zeros((self.hidden_size, 1))
        dc_prev = np.zeros((self.hidden_size, 1))
        
        # Gradient with respect to the concatenated input (h_prev and x_t)
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                dX_t[j] += self.W_f[i, j] * df_t[i] + self.W_i[i, j] * di_t[i] + \
                   self.W_c[i, j] * dc_tilde[i] + self.W_o[i, j] * do_t[i]
            for j in range(self.hidden_size):
                dh_prev[j] += self.W_f[i, j + self.input_size] * df_t[i] + \
                            self.W_i[i, j + self.input_size] * di_t[i] + \
                            self.W_c[i, j + self.input_size] * dc_tilde[i] + \
                            self.W_o[i, j + self.input_size] * do_t[i] 
        
        return dX_t, dh_prev, dc_prev


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
    
    def train(self, model, X_train, y_train):
        total_loss = 0.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_seq, y_true in zip(X_train, y_train):
                # 1. Initialize cell and hidden states for the start of the sequence
                h = np.zeros(self.hidden_size)
                c = np.zeros(self.hidden_size)
                

                # 2. Forward pass through the sequence length
                for t in range(len(X_seq)):
                    x_t = X_seq[t] # 5 features
                    h, c = self.lstm_cell.forward_pass(x_t, h, c)
                
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

                # 6. LSTM cell backpropagation
                dc = np.zeros(self.hidden_size)
                self.lstm_cell.backward_pass(dh, dc, self.learning_rate)
                
            # Average the loss across all samples in the dataset
            epoch_loss /= len(X_train)
            total_loss = epoch_loss

            # 7. Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.6f}")

        return total_loss

    
    def predict(self, features):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        h, c = self.lstm_cell.forward_pass(features, h, c)

        output = self.W_y @ h + self.b_y
        return output[0]
        


def test_overfitting():
    # 1. Setup: 3 inputs, 2 hidden neurons, 1 output
    network = LSTMNetwork(input_size=3, hidden_size=2, output_size=1, learning_rate=0.1, epochs=200)
    
    # 2. Dummy data: A single sequence of 5 time steps
    X_sample = np.random.randn(5, 3) # 5 steps, 3 features
    y_sample = np.array([0.8])       # Target
    
    # Wrap in list so it matches the expected iterable structure
    X_train = [X_sample]
    y_train = [y_sample]

    print(X_train)
    print(y_train)
    
    #3. Train
    print("Starting sanity check (loss should decrease)...")
    network.train(network, X_train, y_train)
    
    #4. Predict
    final_pred = network.predict(X_sample[-1])
    print(f"Target: {y_sample}, Prediction: {final_pred}")

if __name__ == "__main__":
    test_overfitting()
    
