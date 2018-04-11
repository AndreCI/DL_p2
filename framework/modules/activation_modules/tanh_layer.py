from framework.modules.activation_modules.activation_module import ActivationModule

class TanhLayer(ActivationModule):
    '''
    A layer which allows the use of the activation function tanh. This is supposed to be called after an instance of a
    DenseLayer. See dense_layer.
    '''
    def forward(self, input):
        return input.tanh()

    def backward(self, gradient):
        return 1.0/(gradient.cosh()**2)