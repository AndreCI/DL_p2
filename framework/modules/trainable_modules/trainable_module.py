from framework.modules.module import Module

class TrainableModule(Module):
    def forward(self, input):
        raise NotImplementedError()

    def backward(self, gradient):
        raise NotImplementedError()

    def compute_gradient(self, input, error):
        raise NotImplementedError()

    def apply_gradient(self, *gradients, learning_rate):
        raise NotImplementedError()

    def param(self):
        return []