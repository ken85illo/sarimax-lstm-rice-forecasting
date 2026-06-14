import numpy as np
from utils import sigmoid_function, sigmoid_derivative, tanh_function, tanh_derivative, RNG

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        glorot_scale =  np.sqrt(6.0 / ((hidden_size + input_size) + hidden_size)) # Glorot
        shape = (hidden_size, hidden_size + input_size)

        # Forget gate
        self.W_f = RNG.uniform(-glorot_scale, glorot_scale, shape)
        self.b_f = np.zeros(hidden_size)   # initialised to 1 to encourage remembering early on

        # Input gate
        self.W_i = RNG.uniform(-glorot_scale, glorot_scale, shape)
        self.b_i = np.zeros(hidden_size)

        # Cell-state candidate
        self.W_c = RNG.uniform(-glorot_scale, glorot_scale, shape)
        self.b_c = np.zeros(hidden_size)

        # Output gate
        self.W_o = RNG.uniform(-glorot_scale, glorot_scale, shape)
        self.b_o = np.zeros(hidden_size)

        self.reset_gradients()

    # == Forward Pass ==
    def forward_pass(self, x_t, h_prev, c_prev):
        x_t = x_t.copy()
        h_prev = h_prev.copy()
        c_prev = c_prev.copy()

        # Initial step of concatenating input and previous hidden state
        X_t = np.concatenate((h_prev, x_t), axis=0)

        # Forward pass formulas
        f_t = sigmoid_function(self.W_f @ X_t + self.b_f)
        i_t = sigmoid_function(self.W_i @ X_t + self.b_i)
        c_tilde = tanh_function(self.W_c @ X_t + self.b_c)
        c_t = i_t * c_tilde + f_t * c_prev
        o_t = sigmoid_function(self.W_o @ X_t + self.b_o)
        h_t = o_t * tanh_function(c_t)


        state = {
            "x_t":     x_t,
            "h_prev":  h_prev,
            "c_prev":  c_prev,
            "f_t":     f_t,
            "i_t":     i_t,
            "c_tilde": c_tilde,
            "c_t":     c_t,
            "o_t":     o_t,
        }

        # Returns new hidden and cell states to be used in the next pass
        return h_t, c_t, state

    # == Backward Pass ==
    def backward_pass(self, dh_next, dc_next, state):
        x_t = state["x_t"]
        h_prev = state["h_prev"]
        c_prev = state["c_prev"]
        f_t = state["f_t"]
        i_t = state["i_t"]
        c_tilde = state["c_tilde"]
        c_t = state["c_t"]
        o_t = state["o_t"]

        # Standard Gradient Descent
        tanh_c_t = tanh_function(c_t)
        do_t = dh_next * tanh_c_t * sigmoid_derivative(o_t)
        dc_t = dh_next * o_t * tanh_derivative(tanh_c_t) + dc_next
        di_t = dc_t * c_tilde * sigmoid_derivative(i_t)
        dc_tilde = dc_t * i_t * tanh_derivative(c_tilde)
        df_t = dc_t * c_prev * sigmoid_derivative(f_t)

        # Concats input and hidden state ulit
        X_t = np.concatenate((h_prev, x_t), axis=0)
        
        # Accumulate gradients 
        self.dW_f += np.outer(df_t, X_t)
        self.db_f += df_t
        
        self.dW_i += np.outer(di_t, X_t)
        self.db_i += di_t
        
        self.dW_c += np.outer(dc_tilde, X_t)
        self.db_c += dc_tilde
        
        self.dW_o += np.outer(do_t, X_t)
        self.db_o += do_t

        # Propagate gradients to previous time step
        dh_prev = (
            self.W_f[:, : self.hidden_size].T @ df_t
            + self.W_i[:, : self.hidden_size].T @ di_t
            + self.W_c[:, : self.hidden_size].T @ dc_tilde
            + self.W_o[:, : self.hidden_size].T @ do_t
        )
        dc_prev = dc_t * f_t

        return dh_prev, dc_prev
    
    def reset_gradients(self):
        self.dW_f = np.zeros_like(self.W_f)
        self.db_f = np.zeros_like(self.b_f)
        self.dW_i = np.zeros_like(self.W_i)
        self.db_i = np.zeros_like(self.b_i)
        self.dW_c = np.zeros_like(self.W_c)
        self.db_c = np.zeros_like(self.b_c)
        self.dW_o = np.zeros_like(self.W_o)
        self.db_o = np.zeros_like(self.b_o)
    
    def update_weights(self, learning_rate):
        # Update forget gate
        self.W_f -= learning_rate * self.dW_f
        self.b_f -= learning_rate * self.db_f

        # Update input gate
        self.W_i -= learning_rate * self.dW_i
        self.b_i -= learning_rate * self.db_i

        # Update cell state candidate
        self.W_c -= learning_rate * self.dW_c
        self.b_c -= learning_rate * self.db_c

        # Update output gate
        self.W_o -= learning_rate * self.dW_o
        self.b_o -= learning_rate * self.db_o

        # Reset gradients for the next sequence
        self.reset_gradients()
