from torch import FloatTensor, LongTensor
import numpy as np
import torch

#TODO: test input size and add doc
def linear(input, weights, bias=None):
    if bias is not None:
        return input.mm(weights) + bias
    else:
        return input.mm(weights)



def xavier_initialization(in_size, out_size, use_bias=True):
    weights = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / (in_size))
    if use_bias:
        bias = np.random.randn(out_size).astype(np.float32)
        return torch.from_numpy(weights).type(FloatTensor), torch.from_numpy(bias).type(FloatTensor)
    else:
        return torch.from_numpy(weights).type(FloatTensor), None