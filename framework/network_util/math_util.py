from torch import FloatTensor, LongTensor

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