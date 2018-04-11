from framework.modules.trainable_modules.dense_layer import DenseLayer as Dense
from framework.modules.criterion_modules.mse_layer import MSELayer as MSE
from framework.modules.activation_modules.relu_layer import ReLuLayer as ReLu
from framework.modules.activation_modules.tanh_layer import TanhLayer as Tanh
import framework.modules.sequential
import util.data

import torch
import numpy as np


d1 = Dense(2, 25)
h1 = Dense(25, 25)
h2 = Dense(25, 25)
h3 = Dense(25, 25)
out = Dense(25, 2)

tan = Tanh()
relu = ReLu()

Mse = MSE()

layers = [d1, tan, h1, tan, h2, tan, h3, tan, out, tan, Mse]
model = framework.modules.sequential.Sequential(layers=layers)

train_data = util.data.generate_data()
test_data = util.data.generate_data()

train_x = train_data[:, 0:2]
test_x = train_data[:, 0:2]
train_y = train_data[:, 2:4]
test_y = train_data[:, 2:4]
train_X = torch.from_numpy(train_x)
test_X = torch.from_numpy(test_x)
train_Y = torch.from_numpy(train_y)
test_Y = torch.from_numpy(test_y)

util.data.display_data_set(train_x, train_y[:, 0])
epochs = 1
for i in range(epochs):
    total_loss = 0.0
    for j in range(np.shape(train_y)[0]):
        target = torch.from_numpy(np.array([train_y[j]])).type(torch.FloatTensor)
        train_data = torch.from_numpy(train_x[j]).type(torch.FloatTensor)
        target = target.view(1, 2)
        train_data = train_data.view(1, 2)
       # print(target)
        #print(data)
        loss = model.forward(train_data, target)
        model.backward(target, learning_rate=0.05)
        total_loss+=float(loss)
    print(total_loss/(j+1))
            #print(model.memory[-1])
        #exit()
    total_loss = 0.0
    for j in range(np.shape(test_y)[0]):
        target = torch.from_numpy(np.array([test_y[j]])).type(torch.FloatTensor)
        test_data = torch.from_numpy(test_x[j]).type(torch.FloatTensor)
        target = target.view(1, 2)
        test_data = test_data.view(1, 2)
        # print(target)
        # print(data)
        loss = model.forward(train_data, target)
        total_loss += float(loss)
    print(total_loss / (j + 1))
    print("====")