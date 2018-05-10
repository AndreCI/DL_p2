from framework.initializers.Initializer import Initializer
from torch import FloatTensor

class GaussianInitializer(Initializer):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def initialize(self, input_size, output_size, use_bias=True):
        weights = FloatTensor(input_size, output_size).normal_(self.mean, self.std)
        if use_bias:
            bias = FloatTensor(output_size).normal_(self.mean, self.std)
            return weights, bias
        else:
            return weights, None