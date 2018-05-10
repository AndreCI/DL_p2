from framework.initializers.Initializer import Initializer
from torch import FloatTensor

class XavierInitializer(Initializer):
    def initialize(self, input_size, output_size, use_bias=True):
        weights = FloatTensor(input_size, output_size).normal_(0, 2.0 / (input_size + output_size))
        if use_bias:
            bias = FloatTensor(output_size).normal_(0, 2.0 / (input_size + output_size))
            return weights, bias
        else:
            return weights, None