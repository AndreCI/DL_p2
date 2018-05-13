from framework.initializers.Initializer import Initializer
from torch import FloatTensor

class XavierInitializer(Initializer):
    def initialize(self, input_size, output_size, use_bias=True):
        '''
        Initialize the weights and bias if any accordingly to the xavier initialization
        :param input_size: the first dimension of the weights
        :param output_size: the second dimension of the weights
        :param use_bias: generation or not of any bias
        :return: the initialized weights and bias, if any.
        '''
        weights = FloatTensor(input_size, output_size).normal_(0, 2.0 / (input_size + output_size))
        if use_bias:
            bias = FloatTensor(output_size).normal_(0, 2.0 / (input_size + output_size))
            return weights, bias
        else:
            return weights, None