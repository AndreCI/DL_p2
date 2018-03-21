class Module(object):
    '''
    A simple class to implement other modules
    '''
    def forward(self, *input):
        raise NotImplementedError()

    def backward(self, *gradwrtoutput):
        raise NotImplementedError()

    def param(self):
        return []