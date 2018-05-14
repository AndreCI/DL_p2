import math

from torch import FloatTensor

from framework.initializers.Initializer import Initializer


class HeInitializer(Initializer):
    def initialize(self, input_size, output_size, use_bias=True):
        """
        Initialize the weights and bias, if any, accordingly to the "HE" initilization.
        :param input_size: the first dimension of the weights
        :param output_size: the second dimension of the weights
        :param use_bias: generation or not of any bias
        :return: the initialized weights and bias, if any.
        """
        stddev = math.sqrt(2.0 / (input_size + output_size))
        weights = FloatTensor(input_size, output_size).normal_(0, stddev)
        if use_bias:
            bias = FloatTensor(output_size).normal_(0, stddev)
            return weights, bias
        else:
            return weights, None
