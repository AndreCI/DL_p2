from framework.modules.module import Module
from torch import Tensor as T

class TanhLayer(Module):
    '''
    A layer which allows the use of the activation function tanh. This is supposed to be called after an instance of a
    DenseLayer. See dense_layer.
    '''
    def forward(self, input):
        return T.tanh(input)

    def backward(self, gradient):
        return 1.0/(T.cosh(gradient)**2)