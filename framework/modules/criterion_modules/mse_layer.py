from framework.modules.criterion_modules.criterion_module import CriterionModule

class MSELayer(CriterionModule):
    '''
    This layer takes care of the MSE (mean squared error) between the prediction and the target
    '''
    def forward(self, prediction, target):
        '''
        Compute the forward pass for the current example
        :param prediction: the prediction ŷ, i.e. the output of the model
        :param target: the ground truth
        :return: the cost, or loss using MSE, i.e. 1/2 * (y - ŷ)**2
        '''
        #TODO: test if this works.
        if prediction.size() != target.size():
            print(prediction.size())
            print(target.size())
            raise ValueError()
        return 1/2 * ((target - prediction)**2).sum()

    def backward(self, prediction, target):
        '''
        Compute the derivative of this layer.
        :param prediction: the prediction ŷ, i.e. the output of the model
        :param target: the ground truth
        :return: the derivative of MSE, i.e. y - ŷ
        '''
        return target - prediction

    @property
    def type(self):
        return 'mse'