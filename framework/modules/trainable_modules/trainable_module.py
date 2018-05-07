from framework.modules.module import Module

class TrainableModule(Module):
    '''A simple class to represent different activation modules such as linear, etc.'''
    def forward(self, input):
        '''Forward pass is mandatory'''
        raise NotImplementedError()

    def backward(self, gradient):
        '''Backward pass is mandatory'''
        raise NotImplementedError()

    def compute_gradient(self, input, error):
        '''
        Compute the gradient using the last input and the backpropagated error
        :param input: The previous input
        :param error: The error backpropagated
        '''
        raise NotImplementedError()

    def apply_gradient(self, learning_rate):
        '''Apply the gradient computed previously'''
        raise NotImplementedError()

    def param(self):
        '''
        Utility function to see what are the parameters of this modules
        :return: A list of couple containing (parameters, gradients)
        '''
        return []

    def reset(self, initialization='default'):
        '''
        Utility function to reset the weights of the network.
        :param initialization: The type of initalization to use, such as xavier.
        '''
        raise NotImplementedError()