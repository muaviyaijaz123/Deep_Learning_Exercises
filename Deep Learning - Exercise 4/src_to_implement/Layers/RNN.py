import numpy as np
import copy
from .Base import BaseLayer
from .Sigmoid import Sigmoid
from .TanH import TanH
from .FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size, regularizer=None):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.FC_Hidden_Input = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.FC_Hidden_Output = FullyConnected(self.hidden_size, self.output_size)

        self.hidden_state = np.zeros((self.hidden_size,))
        self.memorize = False

        # lists for layer objects for each batch
        self.tanH_List = []
        self.sigmoid_List = []
        self.FC_Hidden_Input_List = []
        self.FC_Hidden_Output_List = []

        self._optimizer = None
        self.regularizer = regularizer

        self._gradient_weights = None
    def initialize(self, weights_initializer, bias_initializer):
        self.FC_Hidden_Input.initialize(weights_initializer, bias_initializer)
        self.FC_Hidden_Output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        regularization_loss = 0.0

        if self._optimizer.regularizer is not None:
            regularization_loss += self._optimizer.regularizer.norm(self.FC_Hidden_Input.weights)
            regularization_loss += self._optimizer.regularizer.norm(self.FC_Hidden_Output.weights)

        return regularization_loss

    def norm(self):
        return self.calculate_regularization_loss()

    def forward(self, input_tensor):
        time_steps = input_tensor.shape[0]

        output_tensor = np.zeros((time_steps, self.output_size))

        if not self.memorize:
            self.hidden_state = np.zeros((self.hidden_size,))

        for time in range(time_steps):
            current_input = input_tensor[time].reshape(1, -1)
            hidden_input = self.hidden_state.reshape(1, -1)
            concatenated_input = np.concatenate((current_input, hidden_input), axis=1)

            #TanH Flow
            FC_Hidden_Input_FCN = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
            self.FC_Hidden_Input_List.append(FC_Hidden_Input_FCN)
            self.FC_Hidden_Input_List[time].weights = self.FC_Hidden_Input.weights

            tanH_Layer = TanH()
            self.tanH_List.append(tanH_Layer)

            concatenated_input_tanh = self.FC_Hidden_Input_List[time].forward(concatenated_input)
            self.hidden_state = self.tanH_List[time].forward(concatenated_input_tanh)


            # Sigmoid Flow
            FC_Hidden_Output_FCN = FullyConnected(self.hidden_size, self.output_size)
            self.FC_Hidden_Output_List.append(FC_Hidden_Output_FCN)
            self.FC_Hidden_Output_List[time].weights = self.FC_Hidden_Output.weights

            sigmoid_Layer = Sigmoid()
            self.sigmoid_List.append(sigmoid_Layer)

            hidden_output_state = self.hidden_state.reshape(1, -1)
            hidden_output = self.FC_Hidden_Output_List[time].forward(hidden_output_state)
            output_tensor[time] = self.sigmoid_List[time].forward(hidden_output)

        return output_tensor

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        prev_layer_error = np.zeros((time_steps, self.input_size))

        gradient_next = np.zeros((self.hidden_size,))

        self.FC_Hidden_Input_Total = np.zeros((self.FC_Hidden_Input.weights.shape[0],self.FC_Hidden_Input.weights.shape[1]+1))
        self.FC_Hidden_Output_Total = np.zeros((self.FC_Hidden_Output.weights.shape[0],self.FC_Hidden_Output.weights.shape[1]+1))

        for time in range(time_steps - 1, -1, -1):

            output_error = error_tensor[time].reshape(1, -1)
            sigmoid_layer_error = self.sigmoid_List[time].backward(output_error)
            FC_Hidden_Output_Error = self.FC_Hidden_Output_List[time].backward(sigmoid_layer_error)

            # Calculate gradient for hidden state
            gradient_Hidden = FC_Hidden_Output_Error + gradient_next
            tanh_layer_error = self.tanH_List[time].backward(gradient_Hidden)
            FC_Hidden_Input_Error = self.FC_Hidden_Input_List[time].backward(tanh_layer_error)


            # Ensure correct shapes for accumulation
            self.FC_Hidden_Input_Total += self.FC_Hidden_Input_List[time].gradient_weights
            self.FC_Hidden_Output_Total += self.FC_Hidden_Output_List[time].gradient_weights


            input_gradient, hidden_gradient = np.split(FC_Hidden_Input_Error, [-self.hidden_size], axis=1)

            gradient_next = hidden_gradient
            prev_layer_error[time] = input_gradient

        if self.optimizer:
            self.FC_Hidden_Input.weights = (self.FC_Hidden_Input.optimizer.
                                            calculate_update(self.FC_Hidden_Input.weights,
                                            self.FC_Hidden_Input_Total[:, :-1]))
            
            self.FC_Hidden_Output.weights = (self.FC_Hidden_Output.optimizer
                                             .calculate_update(self.FC_Hidden_Output.weights,
                                            self.FC_Hidden_Output_Total[:, :-1]))

        return prev_layer_error

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

        self.FC_Hidden_Input.optimizer = copy.deepcopy(value)
        self.FC_Hidden_Output.optimizer = copy.deepcopy(value)


    @property
    def weights(self):
        return self.FC_Hidden_Input.weights

    @weights.setter
    def weights(self, weights):
        hidden_input_weights_size = self.FC_Hidden_Input.weights.size
        self.FC_Hidden_Input.weights = weights[:hidden_input_weights_size].reshape(self.FC_Hidden_Input.weights.shape)

    @property
    def gradient_weights(self):
        return self.FC_Hidden_Input_Total

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        input_hidden_gradient_weights_size = self.FC_Hidden_Input.gradient_weights.size
        self.FC_Hidden_Input._gradient_weights = gradient_weights[:input_hidden_gradient_weights_size].reshape(
            self.FC_Hidden_Input.gradient_weights.shape)
