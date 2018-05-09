from torch import FloatTensor, LongTensor
import math

#TODO: test input size and add doc
def linear(input, weights, bias=None):
    if bias is not None:
        return input.mm(weights) + bias
    else:
        return input.mm(weights)

def xavier_initialization(in_size, out_size, use_bias=True):
    weights = FloatTensor(in_size, out_size).normal_(0, 2.0/(in_size+out_size))
    if use_bias:
        bias = FloatTensor(out_size).normal_(0, 2.0/(in_size+out_size))
        return weights, bias
    else:
        return weights, None

def uniform_initialization(in_size, out_size, min=-1.0, max= 1.0, use_bias=True):
    weights = FloatTensor(in_size, out_size).uniform_(min, max)
    if use_bias:
        bias = FloatTensor(out_size).uniform_(min, max)
        return weights, bias
    else:
        return weights, None

def gaussian_initialization(in_size, out_size, mean=0.0, std= 1.0, use_bias=True):
    weights = FloatTensor(in_size, out_size).normal_(mean, std)
    if use_bias:
        bias = FloatTensor(out_size).normal_(mean, std)
        return weights, bias
    else:
        return weights, None

def he_initialization(in_size, out_size,use_bias=True):
    stddev = math.sqrt(2.0/(in_size + out_size))
    weights = FloatTensor(in_size, out_size).normal_(0,stddev)
    if use_bias:
        bias = FloatTensor(out_size).normal_(0,stddev)
        return weights, bias
    else:
        return weights, None