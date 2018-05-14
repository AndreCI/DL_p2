class Module(object):
    """
    A simple class to implement and combine layers.
    """

    def forward(self, *input):
        """Each module must have a forward pass"""
        raise NotImplementedError()

    def backward(self, *gradwrtoutput):
        """Each module must have a backward pass"""
        raise NotImplementedError()

    def param(self):
        """Each module must have a param method"""
        return []

    def reset(self):
        pass
