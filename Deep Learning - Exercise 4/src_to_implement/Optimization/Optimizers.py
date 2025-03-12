import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self._learning_rate = learning_rate

    def calculate_update(self,  weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularizer_gradient = 0

            # Update weights
        new_weights_tensor = weight_tensor - self._learning_rate * (gradient_tensor + regularizer_gradient)
        return new_weights_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum):
        super().__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularizer_gradient = 0

        self._velocity = self._momentum * self._velocity - self._learning_rate * gradient_tensor

        new_weights_tensor = weight_tensor + self._velocity - (self._learning_rate * regularizer_gradient)
        return new_weights_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate,mu, rho):
        super().__init__()
        self._learning_rate = learning_rate
        self._beta1 = mu
        self._beta2 = rho
        self._epsilon = 1e-8
        self._v = 0
        self._r = 0
        self._k = 0


    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularizer_gradient = 0

        self._k += 1
        self._v = self._beta1 * self._v + (1 - self._beta1) * gradient_tensor
        self._r = self._beta2 * self._r + (1 - self._beta2) * np.square(gradient_tensor)
        v_hat = self._v / (1 - np.power(self._beta1, self._k))
        r_hat = self._r / (1 - np.power(self._beta2, self._k))

        new_weights_tensor = weight_tensor - (self._learning_rate * v_hat / (np.sqrt(r_hat) + self._epsilon)) - (self._learning_rate * regularizer_gradient)
        return new_weights_tensor

