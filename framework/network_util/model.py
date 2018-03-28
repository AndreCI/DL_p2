from framework.modules.module import Module

class Model(Module):
    def __init__(self, layers):
        self.layers = layers
        self.memory = []

    def forward(self, input, target):
        fwd = self.layers[0].forward(input)
        self.memory = [fwd]
        for i in range(1, len(self.layers) - 1):
            fwd = self.layers[i].forward(fwd)
            self.memory.append(fwd)
        fwd = self.layers[-1].forward(fwd, target)
        self.memory.append(fwd)
        return fwd

    def backward(self, input, target):
        error = self.layers[-1].backward(input, target)
        for i in range(len(self.layers) - 1, 1, -1):
            bwd = self.layers[i].backward(bwd)
        raise NotImplementedError()

    def add_layer(self, layer):
        self.layers.append(layer)