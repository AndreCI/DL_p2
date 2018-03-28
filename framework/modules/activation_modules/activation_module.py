from framework.modules.module import Module

class ActivationModule(Module):
    def forward(self, input):
        raise NotImplementedError()

    def backward(self, gradwrtoutput):
        raise NotImplementedError()