from framework.modules.activation_modules.activation_module import ActivationModule

class SigmoidLayer(ActivationModule):
    #TODO: Add doc

    def forward(self, input):
        return input.sigmoid()

    def backward(self, gradient):
        return gradient.sigmoid() * (1 - gradient.sigmoid())

    @property
    def type(self):
        return 'sigmoid'