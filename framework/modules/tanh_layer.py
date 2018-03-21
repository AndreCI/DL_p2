from framework.modules.module import Module
import torch.Tensor as T


class TanhLayer(Module):

    def forward(self, *input):
        return T.tanh(input)

    def backward(self, *gradwrtoutput):
        pass