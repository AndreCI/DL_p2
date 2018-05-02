from framework.modules.module import Module
from framework.modules.trainable_modules.trainable_module import TrainableModule as Trainable
from framework.modules.activation_modules.activation_module import ActivationModule as Activation
from framework.modules.criterion_modules.criterion_module import CriterionModule as Criterion

import torch

class Sequential(Module):
    '''
    A Sequential module which can link multiple dense layers with one another.
    '''
    def __init__(self, layers):
        if not isinstance(layers[-1], Criterion):
            raise AttributeError('Last layer should be a CriterionModule.')
        self.layers = layers
        self.memory = []

    def forward(self, input, target):
        '''
        Forward pass. Returns a couple of tensor containing (prediction, loss)
        :param input: the input data
        :param target: the target, used to compute the loss
        :return: the loss
        '''
        self.memory = [input]
        fwd = self.layers[0].forward(input)
        for i in range(1, len(self.layers) - 1):
            self.memory.append(fwd)
            fwd = self.layers[i].forward(fwd)
        self.memory.append(fwd)
        fwd = self.layers[-1].forward(fwd, target)
        return fwd, self.memory[-1].max(1) #TODO: return only loss?

    def backward(self, target, learning_rate=0.05):
        '''
        Backward pass. Compute the gradient at each layer and apply it to the weights.
        :param target: The target, used to compute the loss
        :param learning_rate: The factor with which the gradient is multiplied before being applied to the the weights
        '''
        error = self.layers[-1].backward(self.memory[-1], target)
        for i in range(len(self.layers) - 2, -1, -1):
            c_layer = self.layers[i]
            if isinstance(c_layer, Trainable):
                next_error = c_layer.backward(error)
                c_layer.compute_gradient(self.memory[i], error)
                c_layer.apply_gradient(learning_rate)
                error = next_error
            elif isinstance(c_layer, Activation):
                slope = c_layer.backward(self.memory[i])
                error = error * slope
            else:
                raise AttributeError('The layers inside the network should be of type TrainableModule or ActivationModule.')
