from .Base import BaseLayer
import numpy as np


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation
    def backward(self, error_tensor):
        activ_derivative = 1 - (self.activation * self.activation)
        return error_tensor * activ_derivative
