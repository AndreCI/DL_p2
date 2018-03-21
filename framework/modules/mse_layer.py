from framework.modules.module import Module
from torch import Tensor as T

class MSELayer(Module):
    '''
    This layer takes care of the MSE (mean squared error) between the prediction and the target
    '''
    def forward(self, prediction, target):
        #TODO: test if size of pred & target are the same.
        return T.sum(prediction - target)**2

    def backward(self, *gradwrtoutput):
        pass

