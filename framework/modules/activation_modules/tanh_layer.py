from framework.modules.activation_modules.activation_module import ActivationModule

class TanhLayer(ActivationModule):
    '''
    A layer which represent a tanh activation.
    '''
    def forward(self, input):
        '''
        Compute the forward pass for this module.
        :param input: the tensor to which apply this activation function
        :return: tanh(input)
        '''
        return input.tanh()

    def backward(self, gradient):
        '''
        Compute the backward pass for this module.
        :param gradient: the error computed by a previous layer
        :return: the derivative of tanh, i.e. 1/cosh²(x)
        '''
        return 1.0/(gradient.cosh()**2)