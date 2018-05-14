from framework.initializers.he_initializer import HeInitializer
from framework.modules.trainable_modules.trainable_module import TrainableModule


class DenseLayer(TrainableModule):
    def __init__(self, in_features, out_features, use_bias=True, initializer=HeInitializer()):
        """
        A simple fully connected layer. There is no activation function, as we consider activation as a layer in and
        on itself. See tanh_layer or relu_layer.
        :param in_features: integer, the dimensionality of the input space
        :param out_features: integer, the dimensionality of the output space
        :param use_bias: boolean, whether the layer uses bias
        """
        self.inputs = in_features
        self.units = out_features
        self.use_bias = use_bias
        self.initializer = initializer
        self.weights, self.bias = self.initializer.initialize(self.inputs, self.units, self.use_bias)
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        """
        Compute the forward pass without an activation function, i.e. XW + B if use_bias is True
        :param input: the current example or output of the previous layer
        :return: XW (+ B)
        """
        if self.use_bias:
            return input.mm(self.weights) + self.bias
        else:
            return input.mm(self.weights)

    def backward(self, gradient):
        """
        Compute the backward pass for this dense layer.
        :param gradient:
        :return:
        """
        return gradient.mm(self.weights.t())

    def compute_gradient(self, input, error):
        """
        Compute the gradient for this dense layer
        :param input: The input for the related error.
        :param error: The backpropagated error until here.
        """
        self.weights_gradient = input.t() * error
        if self.use_bias:
            self.bias_gradient = error.sum()

    def apply_gradient(self, learning_rate):
        """
        Apply the gradient computed previously.
        :param learning_rate: The weights to apply to the gradient.
        """
        self.weights += self.weights_gradient * learning_rate
        if self.use_bias:
            self.bias += self.bias_gradient * learning_rate

    def param(self):
        """
        Useful for debbuging.
        :return: A list of couple, containing the parameters and their gradients.
        """
        return [(self.weights, self.weights_gradient), (self.bias, self.bias_gradient)]

    def reset(self, initializer=None):
        """
        Reset the weights and bias if any accordingly to the initializer given as arg. If it's None, the module will use
        the initializer used during its construction.
        :param initializer: An initilizer object
        """
        if initializer is not None:
            self.initializer = initializer
        self.weights, self.bias = self.initializer.initialize(self.inputs, self.units, self.use_bias)
        self.weights_gradient = None
        self.bias_gradient = None

    @property
    def type(self):
        return 'dense'
