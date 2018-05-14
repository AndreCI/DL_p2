from framework.modules.activation_modules.activation_module import ActivationModule


class SigmoidLayer(ActivationModule):
    # TODO: Add doc

    def forward(self, input):
        """
        Compute the forward pass for this module
        :param input: a tensor
        :return: sigmoid of the input
        """
        return input.sigmoid()

    def backward(self, gradient):
        """
        Compute the backward pass for this module
        :param gradient: the tensor to which apply the backward pass
        :return: the derivative of sigmoid applied to the input
        """
        return gradient.sigmoid() * (1 - gradient.sigmoid())

    @property
    def type(self):
        return 'sigmoid'
