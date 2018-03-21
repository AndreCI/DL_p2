class Module(object):
    '''
    A simple class to implement and combine layers.
    '''
    def forward(self, *input):
        raise NotImplementedError()

    def backward(self, *gradwrtoutput):
        raise NotImplementedError()

    def param(self):
        return []