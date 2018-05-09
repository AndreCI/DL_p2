
class Optimizer(object):
    '''
    A simple abstract class to take care of the optimization process, i.e. learning.
    '''
    def __init__(self):
        self.type = 'abstract'

    def step(self):
        raise NotImplementedError()