import numpy as np
class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha
    def calculate_gradient(self, weights):
        gradient = self.alpha * weights
        return gradient

    def norm(self, weights):
        norm_loss = self.alpha * np.sum(weights**2)
        return norm_loss


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate_gradient(self, weights):
        gradient = self.alpha * np.sign(weights)
        return gradient

    def norm(self, weights):
        norm_loss = self.alpha * np.sum(np.abs(weights))
        return norm_loss