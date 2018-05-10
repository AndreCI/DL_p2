from framework.initializers.Initializer import Initializer
from torch import FloatTensor

class UniformInitializer(Initializer):

    def __init__(self, minval=-1.0, maxval=1.0):
        self.minval = minval
        self.maxval = maxval

    def initialize(self, input_size, output_size, use_bias=True):
        weights = FloatTensor(input_size, output_size).uniform_(self.minval, self.maxval)
        if use_bias:
            bias = FloatTensor(output_size).uniform_(self.minval, self.maxval)
            return weights, bias
        else:
            return weights, None