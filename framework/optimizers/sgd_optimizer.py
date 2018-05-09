from torch import FloatTensor, LongTensor
from framework.optimizers.optimizer import Optimizer
from framework.modules.trainable_modules.trainable_module import TrainableModule

class SGD_optimizer(Optimizer):
    '''
    A simple class to take care of the optimization process, i.e. learning, with stochastic gradient descent.
    '''
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.type = 'SGD'

        self.old_gradients = [[] for _ in range(len(self.model.layers))]

    def step(self, target):
        for i, l in enumerate(self.model.layers):
            if isinstance(l, TrainableModule):
                current_old_grad = [l.weights_gradient, l.bias_gradient]
                self.old_gradients[i] = current_old_grad
        self.model.backward(target) #compute the gradient for each layer
        for i, l in enumerate(self.model.layers):
            if isinstance(l, TrainableModule):
                current_old_grad = self.old_gradients[i]
                if current_old_grad[0] is not None:
                    l.weights_gradient += current_old_grad[0] * self.momentum
                    if l.use_bias:
                        l.bias_gradient += current_old_grad[1] * self.momentum
                l.apply_gradient(learning_rate=self.learning_rate)



