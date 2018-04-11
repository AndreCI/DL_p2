from framework.modules.activation_modules.activation_module import ActivationModule
from torch import FloatTensor, LongTensor

class ReLuLayer(ActivationModule):

    def forward(self, input):
        '''
        Compute the forward pass for this module.
        :param input: the tensor to which apply this activation function
        :return: max(0, input)
        '''
        #TODO: is this really the best way?
        mask = input > 0 #Compute mask
        mask = mask.type(FloatTensor) #Change type of mask to allow mul
        return input.mul(mask) #mask the input

    def backward(self, gradient):
        '''
        Compute the backward pass for this module.
        :param gradient: the error computed by a previous layer
        :return: the derivative of ReLu, i.e. R'(Z) = 0 if Z<0 else 1
        '''
        mask = gradient > 0
        mask = mask.type(FloatTensor)
        return mask