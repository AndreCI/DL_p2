from framework.modules.module import Module

class CriterionModule(Module):
    def forward(self, prediction, target):
        raise NotImplementedError()

    def backward(self, prediction, target):
        raise NotImplementedError()