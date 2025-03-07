from framework.modules.trainable_modules.trainable_module import TrainableModule
from framework.optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):
    """
    A simple class to take care of the optimization process, i.e. learning, with stochastic gradient descent.
    """

    def __init__(self, model, learning_rate, momentum=0.0):
        """
        Initialization of the SGD optimizer
        :param model: a sequential model
        :param learning_rate: parameter to train the model
        :param momentum: a parameter to specify how much of the old gradient to keep
        """
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.type = 'SGD'

        self.old_gradients = [[] for _ in range(len(self.model.layers))]

    def step(self, target):
        """
        Modify the gradients and apply them to the model
        :param target: the expected result
        """
        for i, l in enumerate(self.model.layers):
            if isinstance(l, TrainableModule):
                current_old_grad = [l.weights_gradient, l.bias_gradient]
                self.old_gradients[i] = current_old_grad
        self.model.backward(target)  # compute the gradient for each layer
        for i, l in enumerate(self.model.layers):
            if isinstance(l, TrainableModule):
                current_old_grad = self.old_gradients[i]
                if current_old_grad[0] is not None:
                    l.weights_gradient += current_old_grad[0] * self.momentum
                    if l.use_bias:
                        l.bias_gradient += current_old_grad[1] * self.momentum
                l.apply_gradient(learning_rate=self.learning_rate)
