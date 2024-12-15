import numpy as np

class Constant:
    def __init__(self, value):
        self.value = value

    def initialize(self,weights_shape,fin_in,fin_out):
        return np.full(weights_shape, self.value)


class UniformRandom:
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def initialize(self, weights_shape, fin_in=None, fin_out=None):
        return np.random.uniform(self.low, self.high, weights_shape)


class Xavier:
    def initialize(self, weights_shape, fin_in, fin_out):
        xavier_form = np.sqrt(2/(fin_in+fin_out))
        return np.random.normal(0,scale = xavier_form,size=weights_shape)


class He:
    def initialize(self, weights_shape, fin_in, fin_out):
        he_form = np.sqrt(2/fin_in)
        return np.random.normal(0,scale = he_form,size=weights_shape)

