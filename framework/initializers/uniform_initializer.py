from torch import FloatTensor

from framework.initializers.Initializer import Initializer


class UniformInitializer(Initializer):

    def __init__(self, minval=-1.0, maxval=1.0):
        """
        Initialization of the uniform initializer
        :param minval: The lower bond of the uniform distribution
        :param maxval: the higher bond of the uniform distribution
        """
        self.minval = minval
        self.maxval = maxval

    def initialize(self, input_size, output_size, use_bias=True):
        """
        Initialize the weights and bias, if any, using a uniform distribution
        :param input_size: the first dimension of the weights
        :param output_size: the second dimension of the weights
        :param use_bias: generation or not of any bias
        :return: the initialized weights and bias, if any.
        """
        weights = FloatTensor(input_size, output_size).uniform_(self.minval, self.maxval)
        if use_bias:
            bias = FloatTensor(output_size).uniform_(self.minval, self.maxval)
            return weights, bias
        else:
            return weights, None
