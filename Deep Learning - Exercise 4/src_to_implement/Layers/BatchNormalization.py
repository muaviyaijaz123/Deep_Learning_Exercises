from .Helpers import compute_bn_gradients
from .Base import BaseLayer
import numpy as np

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.input_tensor = None
        self.error_tensor = None
        self.optimizer = None

        self.trainable = True
        self.channels = channels

        self.weights = None # Gamma
        self.bias = None # Beta

        self.mean = None
        self.var = None
        self.moving_avg_mean = None # Moving avg of mean
        self.moving_avg_var = None # Moving avg of variance

        self.eps = np.finfo(float).eps #smallest value
        self.initialize() # initialize Gamma and beta

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        orig_shape = input_tensor.shape
        if input_tensor.ndim == 4:
            self.input_tensor = self.reformat(input_tensor)
        if self.testing_phase:
            self.mean = self.moving_avg_mean
            self.var = self.moving_avg_var
        else:
            self.mean = np.mean(self.input_tensor, axis=0)
            self.var = np.var(self.input_tensor, axis=0)

            if self.moving_avg_mean is None:
                self.moving_avg_mean = self.mean
            if self.moving_avg_var is None:
                self.moving_avg_var = self.var
            else:
                self.moving_avg_mean = self.calculate_avg_mean(self.mean)
                self.moving_avg_var = self.calculate_avg_var(self.var)

        self.X_Mean = (self.input_tensor - self.mean) / (np.sqrt(self.var + self.eps))
        Y_hat = self.weights * self.X_Mean + self.bias

        if len(orig_shape) == 4:
            Y_hat = self.reformat(Y_hat)

        return Y_hat
    def backward(self, error_tensor):
        orig_shape = error_tensor.shape
        self.error_tensor = error_tensor

        if error_tensor.ndim == 4:
            self.error_tensor = self.reformat(error_tensor)

        self.gradient_weights = np.sum(np.multiply(self.error_tensor, self.X_Mean), axis=0)
        self.gradient_bias = np.sum(self.error_tensor, axis = 0)

        bn_gradients = compute_bn_gradients(self.error_tensor, self.input_tensor, self.weights, self.mean, self.var, self.eps)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        if len(orig_shape) == 4:
            bn_gradients = self.reformat(bn_gradients)

        return bn_gradients

    def calculate_avg_mean(self, mean, momentum=0.8):
            return self.moving_avg_mean * momentum + (mean * (1 - momentum))
    def calculate_avg_var(self, var, momentum=0.8):
            return self.moving_avg_var * momentum + (var * (1 - momentum))
    def reformat(self, tensor):
        if tensor.ndim == 4: # conv tensor
            self.updated_shape = tensor.shape
            N, C, H, W = tensor.shape
            updated_tensor = tensor.transpose(0,2,3,1)
            updated_tensor = updated_tensor.reshape(-1,C)
            return updated_tensor
        else:
            N, C, H, W = self.updated_shape
            updated_tensor = tensor.reshape(N, H, W, C)
            updated_tensor =  updated_tensor.transpose(0,3,1,2)
        return updated_tensor





