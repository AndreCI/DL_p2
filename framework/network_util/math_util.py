from torch import Tensor as T
import torch


#TODO: test input size and add doc
def linear(input, weights, bias=None):
    if bias is not None:
        return torch.mm(input, weights) + bias
    else:
        return torch.mm(input, weights)