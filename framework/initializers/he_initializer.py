from framework.initializers.Initializer import Initializer
from torch import FloatTensor
import math

class HeInitializer(Initializer):
    def initialize(self, input_size, output_size, use_bias=True):
        stddev = math.sqrt(2.0 / (input_size + output_size))
        weights = FloatTensor(input_size, output_size).normal_(0, stddev)
        if use_bias:
            bias = FloatTensor(output_size).normal_(0, stddev)
            return weights, bias
        else:
            return weights, None