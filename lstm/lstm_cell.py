import numpy as np
from utils import sigmoid_function, sigmoid_derivative, tanh_function, tanh_derivative

class LSTMCell:
    RNG_MEAN = 0.0
    RNG_STD_DEV = 0.1
    DEFAULT_SCALE = 0.1

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scale = self.DEFAULT_SCALE

        # I think add tayo here ng set seed
        self.rng = np.random.default_rng()

        shape = (hidden_size, hidden_size + input_size)

        # Forget gate
        self.W_f = self.scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, shape)
        self.b_f = np.ones(hidden_size)   # initialised to 1 to encourage remembering early on

        # Input gate
        self.W_i = self.scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, shape)
        self.b_i = np.zeros(hidden_size)

        # Cell-state candidate
        self.W_c = self.scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, shape)
        self.b_c = np.zeros(hidden_size)

        # Output gate
        self.W_o = self.scale * self.rng.normal(self.RNG_MEAN, self.RNG_STD_DEV, shape)
        self.b_o = np.zeros(hidden_size)

    # == Forward Pass ==
    def forward_pass(self, x_t, h_prev, c_prev):
        self.x_t = x_t
        self.h_prev = h_prev
        self.c_prev = c_prev

        # Initial step of concatenating input and previous hidden state
        X_t = np.concatenate((h_prev, x_t), axis=0)

        # Forward pass formulas
        self.f_t = sigmoid_function(self.W_f @ X_t + self.b_f)
        self.i_t = sigmoid_function(self.W_i @ X_t + self.b_i)
        self.c_tilde = tanh_function(self.W_c @ X_t + self.b_c)
        self.c_t = self.i_t * self.c_tilde + self.f_t * c_prev
        self.o_t = sigmoid_function(self.W_o @ X_t + self.b_o)
        self.h_t = self.o_t * tanh_function(self.c_t)

        # Returns new hidden and cell states to be used in the next pass
        return self.h_t, self.c_t

    # == Backward Pass ==
    def backward_pass(self, dh_next, dc_next, learning_rate, state=None, clip_value=5.0):
        if state is not None:
            self.x_t = state["x_t"]
            self.h_prev = state["h_prev"]
            self.c_prev = state["c_prev"]
            self.f_t = state["f_t"]
            self.i_t = state["i_t"]
            self.c_tilde = state["c_tilde"]
            self.c_t = state["c_t"]
            self.o_t = state["o_t"]

        # Standard Gradient Descent
        do_t = dh_next * tanh_function(self.c_t) * sigmoid_derivative(self.o_t)
        dc_t = dh_next * self.o_t * tanh_derivative(tanh_function(self.c_t)) + dc_next
        di_t = dc_t * self.c_tilde * sigmoid_derivative(self.i_t)
        dc_tilde = dc_t * self.i_t * tanh_derivative(self.c_tilde)
        df_t = dc_t * self.c_prev * sigmoid_derivative(self.f_t)

        # Concats input and hidden state ulit
        X_t = np.concatenate((self.h_prev, self.x_t), axis=0)

        # Gradient clipping (prevents grdient explosion)
        do_t = np.clip(do_t, -clip_value, clip_value)
        di_t = np.clip(di_t, -clip_value, clip_value)
        dc_tilde = np.clip(dc_tilde, -clip_value, clip_value)
        df_t = np.clip(df_t, -clip_value, clip_value)

        # Weight updates
        self.W_f -= learning_rate * np.outer(df_t, X_t)
        self.b_f -= learning_rate * df_t

        self.W_i -= learning_rate * np.outer(di_t, X_t)
        self.b_i -= learning_rate * di_t

        self.W_c -= learning_rate * np.outer(dc_tilde, X_t)
        self.b_c -= learning_rate * dc_tilde

        self.W_o -= learning_rate * np.outer(do_t, X_t)
        self.b_o -= learning_rate * do_t

        # Propagate gradients to previous time step
        dh_prev = (
            self.W_f[:, : self.hidden_size].T @ df_t
            + self.W_i[:, : self.hidden_size].T @ di_t
            + self.W_c[:, : self.hidden_size].T @ dc_tilde
            + self.W_o[:, : self.hidden_size].T @ do_t
        )
        dc_prev = dc_t * self.f_t

        return dh_prev, dc_prev