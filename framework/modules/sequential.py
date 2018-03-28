from framework.modules.module import Module
from framework.modules.activation_modules.activation_module import ActivationModule
from framework.modules.criterion_modules.criterion_module import CriterionModule
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
        errors = [self.layers[-1].backward(self.memory[-1], target)]
        for i in range(len(self.layers) - 2, -1, -1):
            error = self.layers[i].backward(errors[-1]) #Assuming D->...D->M
            errors.append(error)
            w_grad, b_grad = self.layers[i].compute_gradient(self.memory[i], errors[-2])
            self.layers[i].apply_gradient(w_grad, b_grad, 0.1)

    def add_layer(self, layer):
        self.layers.append(layer)