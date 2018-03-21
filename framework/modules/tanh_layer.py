from framework.modules.module import Module
from torch import Tensor as T

class TanhLayer(Module):

    def forward(self, input):
        return T.tanh(input)

    def backward(self, *gradwrtoutput):
        pass