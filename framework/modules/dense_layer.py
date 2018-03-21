from framework.modules.module import Module
import torch
from torch import Tensor as T
from ..network_util.math_util import linear

class DenseLayer(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        '''
        A simple fully connected layer
        :param in_features: integer, the dimensionality of the input space
        :param out_features: integer, the dimensionality of the output space
        :param use_bias: boolean, whether the layer uses bias
        '''
        self.inputs = in_features
        self.units = out_features
        self.use_bias = use_bias

        self.weights = torch.randn(in_features, out_features)#T(in_features, out_features)
        if use_bias:
            self.bias = T(out_features)
        else:
            self.bias = None

    def forward(self, input):
        return linear(input, self.weights, self.bias)

    def backward(self, *gradwrtoutput):
        pass