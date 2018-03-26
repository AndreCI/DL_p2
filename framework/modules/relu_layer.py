from framework.modules.module import Module
from torch import Tensor as T
import torch

class ReLuLayer(Module):

    def forward(self, input):
        '''
        Compute the forward pass for this module.
        :param input: the tensor to which apply this activation function
        :return: max(0, input)
        '''
        #TODO: is this really the best way?
        mask = input > 0 #Compute mask
        mask = mask.type(torch.FloatTensor) #Change type of mask to allow mul
        return torch.mul(input, mask) #mask the input

    def backward(self, gradient):
        '''
        Compute the backward pass for this module.
        :param gradient: the error computed by a previous layer
        :return: the derivative of ReLu, i.e. R'(Z) = 0 if Z<0 else 1
        '''
        mask = gradient > 0
        mask = mask.type(torch.FloatTensor)
        return mask