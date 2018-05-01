from framework.modules.module import Module

class ActivationModule(Module):
    '''A simple class to represent different activation modules such as relu, tanh, etc.'''
    def forward(self, input):
        '''Forward pass'''
        raise NotImplementedError()

    def backward(self, gradwrtoutput):
        '''Backward pass'''
        raise NotImplementedError()