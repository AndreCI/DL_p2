from framework.modules.module import Module
from framework.modules.trainable_modules.trainable_module import TrainableModule
from framework.network_util.math_util import linear, xavier_initialization

class DenseLayer(TrainableModule):
    def __init__(self, in_features, out_features, use_bias=True):
        '''
        A simple fully connected layer. There is no activation function, as we consider activation as a layer in and
        on itself. See tanh_layer or relu_layer.
        :param in_features: integer, the dimensionality of the input space
        :param out_features: integer, the dimensionality of the output space
        :param use_bias: boolean, whether the layer uses bias
        '''
        #TODO: could in/out_features could be something else than integer? +Add param verification.
        self.inputs = in_features
        self.units = out_features
        self.use_bias = use_bias

        self.weights, self.bias = xavier_initialization(in_features, out_features, use_bias)
        self.weights_gradient = None
        self.bias_gradient = None
        #self.weights = torch.randn(in_features, out_features) * 3 #T(in_features, out_features).fill_(0.5) #
        #if use_bias:
        #    self.bias = torch.randn(1, out_features) * 3 #T(out_features).fill_(0.5) #
        #else:
        #    self.bias = None

    def forward(self, input):
        '''
        Compute the forward pass without an activation function, i.e. XW + B if use_bias is True
        :param input: the current example or output of the previous layer
        :return: XW (+ B)
        '''
        return linear(input, self.weights, self.bias)

    def backward(self, gradient):
        '''
        Compute the backward pass for this dense layer.
        :param gradient:
        :return:
        '''
        return gradient.mm(self.weights.t())

    def compute_gradient(self, input, error):
        '''
        Compute the gradient for this dense layer
        :param input: The input for the related error.
        :param error: The backpropagated error until here.
        '''
        self.weights_gradient = input.t().mm(error)
        self.bias_gradient = error.sum()

    def apply_gradient(self, learning_rate):
        '''
        Apply the gradient computed previously.
        :param learning_rate: The weights to apply to the gradient.
        '''
        self.weights -= self.weights_gradient * learning_rate
        self.bias -= self.bias_gradient * learning_rate

    def param(self):
        '''
        Useful for debbuging.
        :return: A list of couple, containing the parameters and their gradients.
        '''
        return [(self.weights, self.weights_gradient), (self.bias, self.bias_gradient)]