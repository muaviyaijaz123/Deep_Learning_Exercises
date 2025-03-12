from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        flatten_arr = input_tensor.reshape(self.input_shape[0], -1)
        return flatten_arr

    def backward(self, error_tensor):
        pooling_layer_reshaped_arr = error_tensor.reshape(self.input_shape)
        return pooling_layer_reshaped_arr
