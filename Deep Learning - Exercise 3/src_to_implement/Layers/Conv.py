from .Base import BaseLayer
import numpy as np
import math
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        conv_matrix_weights_shape = (self.num_kernels, *self.convolution_shape)
        self.weights = np.random.uniform(size=conv_matrix_weights_shape)
        self.bias = np.random.uniform(size=(self.num_kernels))  # bias: one for every kernel

        self.input_tensor = None

        self.conv_weights_gradient = None
        self.conv_biases_gradient = None

        self.weights_optimizer = None
        self.bias_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)  # [input channels × kernel height × kernel width]
        fan_out = np.prod(
            self.convolution_shape[1:]) * self.num_kernels  # [output channels × kernel height × kernel width]
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_1d_convolution_input = input_tensor.ndim == 3

        if is_1d_convolution_input:

            batch_size = input_tensor.shape[0]
            input_width = input_tensor.shape[2]

            kernels_num = self.weights.shape[0]
            conv_width = self.weights.shape[2]

            # Padding
            pad_total = conv_width - 1
            left_pad_width = pad_total // 2
            right_pad_width = pad_total - left_pad_width

            # Zero Padding
            padded_input = np.pad(
                input_tensor,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (left_pad_width, right_pad_width)
                ),mode='constant', constant_values=0)

            # Calculate the width of the output tensor
            adjusted_width = input_width + pad_total - conv_width
            output_width = (adjusted_width // self.stride_shape[0]) + 1

            # Initialize the tensor to hold convolution results
            convolution_result_shape = (batch_size, kernels_num, output_width)
            convoluted_output = np.zeros(convolution_result_shape)

            for batch_idx in range(batch_size):  # Iterate over each batch
                for kernel_idx in range(kernels_num):  # Iterate over each kernel
                    for width_idx in range(output_width):  # Iterate over the output width

                        conv_range = slice(width_idx * self.stride_shape[0],width_idx * self.stride_shape[0] + conv_width)
                        conv_window = padded_input[batch_idx, :, conv_range]

                        result = np.sum(conv_window * self.weights[kernel_idx])
                        convoluted_output[batch_idx, kernel_idx, width_idx] = result + self.bias[kernel_idx]

        else:  # 2D convolution

            batch_size = input_tensor.shape[0]
            input_height = input_tensor.shape[2]
            input_width = input_tensor.shape[3]

            kernels_num = self.weights.shape[0]
            conv_height = self.weights.shape[2]
            conv_width = self.weights.shape[3]

            # Padding
            total_pad_height = conv_height - 1
            total_pad_width = conv_width - 1

            top_pad_height = total_pad_height // 2
            bottom_pad_height = total_pad_height - top_pad_height
            left_pad_width= total_pad_width // 2
            right_pad_width = total_pad_width - left_pad_width

            # Zero Padding
            padded_input = np.pad(
                input_tensor,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (top_pad_height, bottom_pad_height),
                    (left_pad_width, right_pad_width)
                ),mode='constant',constant_values=0)

            # Calculate the height and width of the output tensor

            adjusted_height = input_height + total_pad_height - conv_height
            adjusted_width = input_width + total_pad_width - conv_width

            output_height = (adjusted_height // self.stride_shape[0]) + 1
            output_width = (adjusted_width // self.stride_shape[1]) + 1

            # Initialize the tensor to hold convolution results

            convolution_result_shape = (batch_size, kernels_num, output_height, output_width)
            convoluted_output = np.zeros(convolution_result_shape)

            for batch_idx in range(batch_size):
                for kernel_idx in range(kernels_num):
                    for height_idx in range(output_height):
                        for width_idx in range(output_width):
                            conv_height_slice = slice(height_idx * self.stride_shape[0],   height_idx * self.stride_shape[0] + conv_height)
                            conv_width_slice = slice(width_idx * self.stride_shape[1],width_idx * self.stride_shape[1] + conv_width)

                            conv_window = padded_input[batch_idx, :, conv_height_slice, conv_width_slice]
                            result = np.sum(conv_window * self.weights[kernel_idx])
                            convoluted_output[batch_idx, kernel_idx, height_idx, width_idx] = result + self.bias[kernel_idx]

        return convoluted_output

    def backward(self, error_tensor):

        self.conv_weights_gradient = np.zeros(self.weights.shape)  # gradient weight
        self.conv_biases_gradient = np.zeros(self.bias.shape)  # gradient bias

        is_1d_convolution_input = error_tensor.ndim == 3

        if is_1d_convolution_input:
            batch_size = error_tensor.shape[0]
            kernels_num = error_tensor.shape[1]
            error_width = error_tensor.shape[2]

            conv_width = self.weights.shape[2]

            pad_width = (conv_width - 1) // 2
            padded_input = np.pad(self.input_tensor,
                                  ((0, 0), (0, 0),
                                   (pad_width, pad_width)), mode='constant', constant_values=0)

            output_error = np.zeros(padded_input.shape, dtype=padded_input.dtype)

            for batch in range(batch_size):
                for kernel in range(kernels_num):
                    self.conv_biases_gradient[kernel] += np.sum(error_tensor[batch, kernel])  # gradient bias

                    for width in range(error_width):
                        width_slice = slice(width * self.stride_shape[0], width * self.stride_shape[0] + conv_width)
                        window = padded_input[batch, :, width_slice]

                        self.conv_weights_gradient[kernel] += np.sum(window * error_tensor[batch, kernel, width])  # gradient weight
                        output_error[batch, :, width_slice] += self.weights[kernel] * error_tensor[batch, kernel, width]

            if pad_width > 0:
                start_idx = pad_width
                end_idx = -pad_width
                output_error = output_error[:, :, start_idx:end_idx]


        else:
            batch_size = error_tensor.shape[0]
            kernels_num = error_tensor.shape[1]
            error_height = error_tensor.shape[2]
            error_width = error_tensor.shape[3]

            conv_height = self.weights.shape[2]
            conv_width = self.weights.shape[3]

            pad_height = (conv_height - 1) // 2
            pad_width = (conv_width - 1) // 2

            padded_input = np.pad(self.input_tensor,
                                  ((0, 0),
                                   (0, 0),
                                   (pad_height, pad_height), (pad_width, pad_width)),
                                  mode='constant',constant_values=0)

            output_error = np.zeros(padded_input.shape, dtype=padded_input.dtype)

            for batch_idx in range(batch_size):
                for kernel_idx in range(kernels_num):
                    self.conv_biases_gradient[kernel_idx] += np.sum(error_tensor[batch_idx, kernel_idx])

                    for height_idx in range(error_height):
                        for width_idx in range(error_width):

                            height_start = height_idx * self.stride_shape[0]
                            height_end = height_start + conv_height
                            width_start = width_idx * self.stride_shape[1]
                            width_end = width_start + conv_width

                            if height_end > padded_input.shape[2] or width_end > padded_input.shape[3]:
                                continue

                            height_slice = slice(height_start, height_end)
                            width_slice = slice(width_start, width_end)

                            conv_window = padded_input[batch_idx, :, height_slice, width_slice]

                            self.conv_weights_gradient[kernel_idx] += conv_window * error_tensor[
                                batch_idx, kernel_idx, height_idx, width_idx]

                            output_error[batch_idx, :, height_slice, width_slice] += self.weights[kernel_idx] * \
                                                                                     error_tensor[
                                                                                         batch_idx, kernel_idx,
                                                                                         height_idx, width_idx]

            output_error = output_error[:, :, pad_height:-pad_height if pad_height > 0 else None,
                           pad_width:-pad_width if pad_width > 0 else None]


        if self.weights_optimizer:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.conv_weights_gradient)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.conv_biases_gradient)

        return output_error

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self.conv_weights_gradient

    @property
    def gradient_bias(self):
        return self.conv_biases_gradient
