import numpy as np
from .Base import BaseLayer
class Pooling(BaseLayer):
    def __init__(self,stride_shape,pooling_shape):
        super().__init__()
        self.trainable = False

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

        self.input_shape = None
        self.max_indices = None
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        m = self.pooling_shape[0]
        n = self.pooling_shape[1]

        batches = input_tensor.shape[0]
        channels = input_tensor.shape[1]
        input_length_v = input_tensor.shape[2]
        input_length_h = input_tensor.shape[3]

        vertical_output_length = (input_length_v - m) // self.stride_shape[0] + 1
        horizontal_output_length = (input_length_h - n) // self.stride_shape[1] + 1

        output_tensor = np.zeros((batches, channels, vertical_output_length, horizontal_output_length))
        self.max_indices = np.zeros((batches, channels, vertical_output_length, horizontal_output_length, 2), dtype=int)

        for batch in range(batches):
            for channel in range(channels):
                for vertical_length in range(vertical_output_length):
                    for horizontal_length in range(horizontal_output_length):
                        vertical_start = vertical_length * self.stride_shape[0]
                        horizontal_start = horizontal_length * self.stride_shape[1]

                        vertical_slice = slice(vertical_start, vertical_start + m)
                        horizontal_slice = slice(horizontal_start, horizontal_start + n)

                        pool_slice = input_tensor[batch, channel, vertical_slice, horizontal_slice]

                        max_index = np.unravel_index(np.argmax(pool_slice),
                                                     pool_slice.shape)
                        self.max_indices[batch, channel, vertical_length, horizontal_length] = (
                        vertical_start + max_index[0], horizontal_start + max_index[1])

                        output_tensor[batch, channel, vertical_length, horizontal_length] = np.max(pool_slice)

        return output_tensor
    def backward(self, error_tensor):
        error_tensor_shape = error_tensor.shape
        batches = error_tensor_shape[0]
        channels = error_tensor_shape[1]

        error_length_v = error_tensor_shape[2]
        error_length_h = error_tensor_shape[3]

        # Initialize error output
        error_output = np.zeros(self.input_tensor.shape)

        for batch in range(batches):
            for channel in range(channels):
                for vertical_dimension in range(error_length_v):
                    for horizontal_dimension in range(error_length_h):

                        vertical_position, horizontal_position = self.max_indices[batch, channel, vertical_dimension, horizontal_dimension]
                        error_output[batch, channel, vertical_position, horizontal_position] += error_tensor[batch, channel, vertical_dimension, horizontal_dimension]

        return error_output





