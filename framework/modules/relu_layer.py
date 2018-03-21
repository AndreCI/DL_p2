from framework.modules.module import Module
from torch import Tensor as T
import torch

class ReLuLayer(Module):

    def forward(self, input):
        #TODO: is this really the best way?
        mask = input > 0 #Compute mask
        mask = mask.type(torch.FloatTensor) #Change type of mask to allow mul
        return torch.mul(input, mask) #mask the input

    def backward(self, *gradwrtoutput):
        pass