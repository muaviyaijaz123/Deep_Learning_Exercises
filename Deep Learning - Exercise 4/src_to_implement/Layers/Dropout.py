from .Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.input_tensor = None
        self.probability = probability
        self.testing_phase = False

    def forward(self, input_tensor):

        if self.testing_phase:
            return input_tensor
        else:
            random_values = np.random.rand(*input_tensor.shape)
            self.mask = np.where(random_values > 1 - self.probability, True, False)
            return input_tensor * self.mask / (self.probability)

    def backward(self, error_tensor):

        if not self.testing_phase:
            return error_tensor * self.mask / (self.probability)

        else:
            return error_tensor


