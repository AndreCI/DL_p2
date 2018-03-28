from framework.modules.module import Module
from torch import Tensor as T

class MSELayer(Module):
    '''
    This layer takes care of the MSE (mean squared error) between the prediction and the target
    '''
    def forward(self, prediction, target):
        '''
        Compute the forward pass for the current example
        :param prediction: the prediction ŷ, i.e. the output of the model
        :param target: the ground truth
        :return: the cost, or loss using MSE, i.e. 1/2 * (ŷ - y)**2
        '''
        #TODO: test if size of pred & target are the same.
        return 1/2 * T.sum((prediction - target)**2)

    def backward(self, prediction, target):
        '''
        Compute the derivative of this layer.
        :param prediction: the prediction ŷ, i.e. the output of the model
        :param target: the ground truth
        :return: the derivative of MSE, i.e. ŷ - y
        '''
        return prediction - target

