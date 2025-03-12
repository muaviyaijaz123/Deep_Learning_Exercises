import copy

import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.input_tensor = None
        self.prediction_data = None

        # CNN Exercise-2 change
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

        # Exercise 3 change
        self.reg_loss = 0

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()  # Fetch new batch of data

        for layer in self.layers:
            layer.testing_phase = False
            self.input_tensor = layer.forward(self.input_tensor)
            # if self.optimizer.regularizer is not None:
            #     self.reg_loss += self.optimizer.regularizer.norm(layer.weights)

        self.prediction_data = self.input_tensor

        return self.prediction_data.tolist()

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)  # initialize trainable layers
        self.layers.append(layer)

    def train(self, iterations):

        for i in range(iterations):
            self.forward()
            loss_error = self.loss_layer.forward(self.prediction_data, self.label_tensor)

            self.reg_loss = 0

            if self.optimizer.regularizer is not None:
                for layer in self.layers:
                    if hasattr(layer, 'weights') and layer.weights is not None:
                        self.reg_loss += self.optimizer.regularizer.norm(layer.weights)

            loss_error += self.reg_loss

            self.backward()
            self.loss.append(loss_error)

    def test(self, input_tensor):
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)

        return input_tensor

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
