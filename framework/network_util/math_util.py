from torch import Tensor as T
import torch
import numpy as np

#TODO: test input size and add doc
def linear(input, weights, bias=None):
    if bias is not None:
        return torch.mm(input, weights) + bias
    else:
        return torch.mm(input, weights)


def xavier_initialization(in_size, out_size, use_bias=True):
    weights = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / (in_size))
    if use_bias:
        bias = np.random.randn(out_size).astype(np.float32)
        return torch.from_numpy(weights).type(torch.FloatTensor), torch.from_numpy(bias).type(torch.FloatTensor)
    else:
        return torch.from_numpy(weights).type(torch.FloatTensor), None