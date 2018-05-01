from framework.modules.trainable_modules.dense_layer import DenseLayer as Dense
from framework.modules.criterion_modules.mse_layer import MSELayer as MSE
from framework.modules.activation_modules.relu_layer import ReLuLayer as ReLu
from framework.modules.activation_modules.tanh_layer import TanhLayer as Tanh
import framework.modules.sequential
import util.data
import numpy as np
from torch import FloatTensor, LongTensor
import argparse
from util.configuration import get_args, setup_log

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

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
point_number = opt['point_number']

train_examples, train_targets = util.data.generate_data(points_number=point_number)
test_examples, test_targets = util.data.generate_data(points_number=point_number)


util.data.display_data_set(opt, train_examples, train_targets[:, 0])
epochs = 1
final_tr_loss = 0.0
final_te_loss = 0.0
for i in range(epochs):
    total_loss = 0.0
    for j in range(0, point_number):
        target = train_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        train_data = train_examples[j].view(1, 2)
        loss = model.forward(train_data, target)
        model.backward(target, learning_rate=opt['lr'])
        total_loss+=float(loss)
        message = str('Training loss %3f - iteration %i/%i, epoch %i/%i' %(loss, j, point_number, i, epochs))
        log.info(message)
    final_tr_loss += total_loss/(j+1)
    message = str('Average Training loss %3f - epoch %i/%i' % (total_loss/(j+1), (i+1), epochs))
    log.info(message)
    total_loss = 0.0
    for j in range(0, point_number):
        target = test_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        test_data = test_examples[j].view(1, 2)
        loss = model.forward(train_data, target)
        total_loss += float(loss)
        message = str('Testing loss %3f - iteration %i/%i, epoch %i/%i' % (loss, j, point_number, i, epochs))
        log.info(message)
    message = str('Average Testing loss %3f - epoch %i/%i' % (total_loss / (j + 1), (i+1), epochs))
    log.info(message)
    final_te_loss += total_loss/(j+1)
print("-" * 60)
print("Average Training loss accross all the epochs: %3f" %(final_tr_loss/epochs))
print("Average Testing loss accross all the epochs: %3f" %(final_te_loss/epochs))