from framework.modules.module import Module
from framework.modules.activation_modules.activation_module import ActivationModule
from framework.modules.criterion_modules.criterion_module import CriterionModule
from framework.modules.trainable_modules.trainable_module import TrainableModule as Trainable
from framework.modules.activation_modules.activation_module import ActivationModule as Activable

import numpy as np

class Sequential(Module):
    #TODO: add contition, as user should end with MSE layer
    def __init__(self, layers):
        self.layers = layers
        self.memory = []

    def forward(self, input, target):
        self.memory = [input]
        fwd = self.layers[0].forward(input)
        for i in range(1, len(self.layers) - 1):
            self.memory.append(fwd)
            fwd = self.layers[i].forward(fwd)
        self.memory.append(fwd)
        fwd = self.layers[-1].forward(fwd, target)
        return fwd

    def backward(self, prediction, target):
        error = self.layers[-1].backward(self.memory[-1], target)
        for i in range(len(self.layers) - 2, -1, -1):
            c_layer = self.layers[i]
            if isinstance(c_layer, Trainable):
                next_error = c_layer.backward(error) #Assuming D->...D->M
                w_grad, b_grad = c_layer.compute_gradient(self.memory[i], error)
                error = next_error
                c_layer.apply_gradient(w_grad, b_grad, 0.2)
            elif isinstance(c_layer, Activable):
                slope = c_layer.backward(self.memory[i])
                error = error * slope
            else:
                raise NotImplementedError()

    def add_layer(self, layer):
        self.layers.append(layer)
