# File: Sigmoid.py
# Folder: Layers

import numpy as np
from .Base import BaseLayer  # Assuming Base.py contains the BaseLayer class

class Sigmoid(BaseLayer):
    def __init__(self):
        self.activation = None
        super().__init__()

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-input_tensor))
        return self.activation

    def backward(self, error_tensor):
        gradient = self.activation * (1 - self.activation)
        grad_input = np.multiply(error_tensor, gradient)
        return grad_input