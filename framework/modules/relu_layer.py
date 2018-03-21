from framework.modules.module import Module
from torch import Tensor as T
import torch

class ReLuLayer(Module):

    def forward(self, input):
        return torch.relu(input)

    def backward(self, *gradwrtoutput):
        pass