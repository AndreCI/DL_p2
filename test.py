from framework.modules.trainable_modules.dense_layer import DenseLayer as Dense
from framework.modules.criterion_modules.mse_layer import MSELayer as MSE
from framework.modules.activation_modules.relu_layer import ReLuLayer as ReLu
from framework.modules.activation_modules.tanh_layer import TanhLayer as Tanh
import framework.modules.sequential
import util.data_generation


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

layers = [d1, relu, h1, relu, h2, tan, h3, relu, out, relu, Mse]
model = framework.modules.sequential.Sequential(layers)

data = util.data_generation.generate_data()
x = data[:,0:2]
y = data[:,2]
X = torch.from_numpy(x)
Y = torch.from_numpy(y)

epochs = 50
for i in range(epochs):
    total_loss = 0.0
    for j in range(np.shape(y)[0]):
        target = torch.from_numpy(np.array([y[j]])).type(torch.FloatTensor)
        data = torch.from_numpy(x[j]).type(torch.FloatTensor)
        target = target.view(1, 1)
        data = data.view(1, 2)
        loss = model.forward(data, target)
        model.backward(target)
        total_loss+=float(loss)
        if j%10 ==0:
            print(total_loss/(j+1))
    print(total_loss/np.shape(y)[0])