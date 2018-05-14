import json
import os

from torch import FloatTensor

from framework.modules.activation_modules.activation_module import ActivationModule as Activation
from framework.modules.activation_modules.relu_layer import ReLuLayer
from framework.modules.activation_modules.sigmoid_module import SigmoidLayer
from framework.modules.activation_modules.tanh_layer import TanhLayer
from framework.modules.criterion_modules.criterion_module import CriterionModule as Criterion
from framework.modules.criterion_modules.mse_layer import MSELayer
from framework.modules.module import Module
from framework.modules.trainable_modules.dense_layer import DenseLayer
from framework.modules.trainable_modules.trainable_module import TrainableModule as Trainable


class Sequential(Module):
    """
    A Sequential module which can link multiple dense layers with one another.
    """

    def __init__(self, layers):
        if not isinstance(layers[-1], Criterion):
            raise AttributeError('Last layer should be a CriterionModule.')
        self.layers = layers
        self.memory = []

    def forward(self, input, target):
        """
        Forward pass. Returns a couple of tensor containing (prediction, loss)
        :param input: the input data
        :param target: the target, used to compute the loss
        :return: the loss
        """
        self.memory = [input]
        fwd = self.layers[0].forward(input)
        for i in range(1, len(self.layers) - 1):
            self.memory.append(fwd)
            fwd = self.layers[i].forward(fwd)
        self.memory.append(fwd)
        fwd = self.layers[-1].forward(fwd, target)
        return fwd, self.memory[-1].min(1)  # TODO: return only loss?

    def backward(self, target, learning_rate=0.05):
        """
        Backward pass. Compute the gradient at each layer and apply it to the weights.
        :param target: The target, used to compute the loss
        :param learning_rate: The factor with which the gradient is multiplied before being applied to the the weights
        """
        error = self.layers[-1].backward(self.memory[-1], target)
        for i in range(len(self.layers) - 2, -1, -1):
            c_layer = self.layers[i]
            if isinstance(c_layer, Trainable):
                c_layer.compute_gradient(self.memory[i], error)
                error = c_layer.backward(error)
            elif isinstance(c_layer, Activation):
                slope = c_layer.backward(self.memory[i + 1])
                error = error * slope
            else:
                raise AttributeError(
                    'The layers inside the network should be of type TrainableModule or ActivationModule.')

    def param(self):
        param = []
        for c_layer in self.layers:
            if isinstance(c_layer, Trainable):
                param.extend(c_layer.param())
        return param

    def reset(self):
        """
        Reset each of the layer.s
        """
        for l in self.layers:
            l.reset()

    def save_model(self, name, save_dir, test_acc=0.0):
        """
        Save the current model in a json format
        :param name: the name of the file
        :param save_dir: the path to which save the file
        :param test_acc: additional info to add to the json.
        """
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        data = {'layers': [], 'test_accuracy': test_acc}
        for i, l in enumerate(self.layers):
            current_layer = {}
            if isinstance(l, Trainable):
                current_layer['name'] = 'trainable_' + str(i)
                current_layer['type'] = l.type
                current_layer['in_features'] = l.inputs
                current_layer['out_features'] = l.units
                current_layer['has_bias'] = l.use_bias
                weights = []
                for col in l.weights.numpy():
                    new_col = []
                    for w in col:
                        new_col.append(float(w))
                    weights.append(new_col)
                bias = []
                for b in l.bias.numpy():
                    bias.append(float(b))
                current_layer['weights'] = weights
                current_layer['bias'] = bias
            elif isinstance(l, Activation):
                current_layer['name'] = 'activation_' + str(i)
                current_layer['type'] = l.type
            elif isinstance(l, Criterion):
                current_layer['name'] = 'criterion_' + str(i)
                current_layer['type'] = l.type
            data['layers'].append(current_layer)
        file = os.path.join(save_dir, str(name + '.json'))
        with open(file, 'w', encoding='utf-8') as file:
            json.dump(data, file)

    @staticmethod
    def load_model(name, save_dir):
        """
        A static method to load a model from a json file
        :param name: The name of the file
        :param save_dir: the path of the file
        :return: an already build and trained network
        """
        file = os.path.join(save_dir, str(name + '.json'))
        with open(file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        layers = []
        for l in data['layers']:
            current_layer = None
            if l['type'] == 'dense':
                current_layer = DenseLayer(l['in_features'], l['out_features'], l['has_bias'])
                current_layer.weights = FloatTensor(l['weights'])
                current_layer.bias = FloatTensor(l['bias'])
            elif l['type'] == 'sigmoid':
                current_layer = SigmoidLayer()
            elif l['type'] == 'relu':
                current_layer = ReLuLayer()
            elif l['type'] == 'tanh':
                current_layer = TanhLayer()
            elif l['type'] == 'mse':
                current_layer = MSELayer()
            layers.append(current_layer)
        model = Sequential(layers)
        return model
