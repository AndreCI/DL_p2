from framework.modules.module import Module
from framework.modules.trainable_modules.trainable_module import TrainableModule as Trainable
from framework.modules.activation_modules.activation_module import ActivationModule as Activation
from framework.modules.criterion_modules.criterion_module import CriterionModule as Criterion

import numpy as np

class Sequential(Module):
    #TODO: add contition, as user should end with MSE layer
    def __init__(self, layers):
        if not isinstance(layers[-1], Criterion):
            raise AttributeError('Last layer should be a CriterionModule.')
        self.layers = layers
        self.memory = []

    def forward(self, input, target, mode='train'):
        self.memory = [input]
        fwd = self.layers[0].forward(input)
        for i in range(1, len(self.layers) - 1):
            self.memory.append(fwd)
            fwd = self.layers[i].forward(fwd)
        self.memory.append(fwd)
        fwd = self.layers[-1].forward(fwd, target)
        return fwd

    def backward(self, target):
        error = self.layers[-1].backward(self.memory[-1], target)
        for i in range(len(self.layers) - 2, -1, -1):
            c_layer = self.layers[i]
            if isinstance(c_layer, Trainable):
                next_error = c_layer.backward(error)
                w_grad, b_grad = c_layer.compute_gradient(self.memory[i], error)
                error = next_error
                c_layer.apply_gradient(w_grad, b_grad, 0.05)
            elif isinstance(c_layer, Activation):
                slope = c_layer.backward(self.memory[i])
                error = error * slope
            else:
                raise AttributeError('The layers inside the network should be of type TrainableModule or ActivationModule.')
