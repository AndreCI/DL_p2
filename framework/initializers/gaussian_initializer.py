from torch import FloatTensor

from framework.initializers.Initializer import Initializer


class GaussianInitializer(Initializer):
    def __init__(self, mean=0.0, std=1.0):
        """
        Initialization of the Gaussian Initiliazer.
        :param mean: The mean to which the gaussian distribution will be centered
        :param std: The std that the gaussian distribution will use
        """
        self.mean = mean
        self.std = std

    def initialize(self, input_size, output_size, use_bias=True):
        """
        Initialize weights using a gaussian distribution
        :param input_size: the first dimension of the weights
        :param output_size: the second dimension of the weights
        :param use_bias: generation or not of any bias
        :return: the initialized weights and bias, if any.
        """
        weights = FloatTensor(input_size, output_size).normal_(self.mean, self.std)
        if use_bias:
            bias = FloatTensor(output_size).normal_(self.mean, self.std)
            return weights, bias
        else:
            return weights, None
