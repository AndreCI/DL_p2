from framework.modules.module import Module

class CriterionModule(Module):
    '''A simple class to represent different criterion modules such as mse, mae, etc.'''
    def forward(self, prediction, target):
        '''Forward pass'''
        raise NotImplementedError()

    def backward(self, prediction, target):
        '''Backward pass'''
        raise NotImplementedError()

    @property
    def type(self):
        raise NotImplementedError()