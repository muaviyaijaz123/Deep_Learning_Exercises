import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._weights = np.random.uniform(0, 1, (output_size, input_size + 1))
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = None

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        biased_input = np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))])
        self.input_tensor = biased_input
        return np.dot(biased_input, np.transpose(self._weights))

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(np.transpose(self.input_tensor), error_tensor)
        self._gradient_weights = np.transpose(self._gradient_weights)
        grad_input = np.dot(error_tensor, self._weights[:, :-1])

        bias_gradients = np.sum(error_tensor, axis=0, keepdims=True)
        self._gradient_weights = np.hstack([self._gradient_weights, bias_gradients.T])

        if self._optimizer:
            self._weights = self._optimizer.calculate_update(self._weights, self._gradient_weights[:, :-1])
        return grad_input

    # only change of ex2
    def initialize(self, weights_initializer, bias_initializer):
        self._weights[:, :-1] = weights_initializer.initialize((self.output_size, self.input_size), self.input_size, self.output_size) # initialize(weights_shape, fan_in, fan_out)
        self._weights[:, -1] = bias_initializer.initialize((self.output_size,), self.input_size, self.output_size) # add initialized bias in weight matrix

    def norm(self):
        if self._optimizer and self._optimizer.regularizer:
            return self._optimizer.regularizer.norm(self._weights)
        return 0