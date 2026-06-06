import numpy as np
from utils import sigmoid_function, sigmoid_derivative, tanh_function, tanh_derivative

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

        # # Gradient of the output gate
        # do_t  = dh_next * tanh_function(self.c_t) * sigmoid_derivative(self.o_t)

        # # Gradient of the cell state
        # dc_t = dh_next * self.o_t * tanh_derivative(self.c_t) + dc_next

        # # Gradient of the input gate
        # di_t = dc_t * self.c_tilde * sigmoid_derivative(self.i_t)

        # # Gradient of the cell state candidate
        # dc_tilde = dc_t * self.i_t * tanh_derivative(self.c_tilde)

        # # Gradient of the forget gate
        # df_t = dc_t * self.c_prev * sigmoid_derivative(self.f_t)

        do_t  = dh_next * tanh_function(self.c_t) * (self.o_t * (1 - self.o_t))
        dc_t  = dh_next * self.o_t * tanh_derivative(self.c_t) + dc_next
        di_t  = dc_t * self.c_tilde * (self.i_t * (1 - self.i_t))
        df_t  = dc_t * self.c_prev * (self.f_t * (1 - self.f_t))
        dc_tilde = dc_t * self.i_t * (1 - self.c_tilde**2)

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
