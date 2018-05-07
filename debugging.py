from framework.modules.trainable_modules.dense_layer import DenseLayer as Dense
from framework.modules.criterion_modules.mse_layer import MSELayer as MSE
from framework.modules.activation_modules.relu_layer import ReLuLayer as ReLu
from framework.modules.activation_modules.tanh_layer import TanhLayer as Tanh
from framework.modules.activation_modules.sigmoid_module import SigmoidLayer as Sigmoid
import framework.modules.sequential
import util.data
from torch import FloatTensor, LongTensor
from util.configuration import get_args, setup_log
import argparse
import torch

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

point_number = 10000
train_examples, train_targets = util.data.generate_toy_data(point_number)# util.data.generate_data(points_number=point_number)
test_examples, test_targets = util.data.generate_toy_data() #generate_data(points_number=point_number)

util.data.display_data_set(opt, train_examples, train_targets[:, 0])

def ps(x):
    a = 1+1
    #print(x.size())

d1 = Dense(2, 2, True)
h1 = Dense(2, 2, True)
out = Dense(2, 2, True)

tan = Sigmoid()
relu = ReLu()

Mse = MSE()
predictions = []
total_loss = 0.0
final_tr_acc = 0.0
final_tr_loss = 0.0
learning_rate = 0.1
for train_test in range(2):
    predictions = []
    total_loss = 0.0
    final_tr_acc = 0.0
    final_tr_loss = 0.0
    learning_rate = 0.1
    for j in range(0, point_number):
        target = train_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        train_data = train_examples[j].view(1, 2)
        m0 = train_data
        ps(m0)
        m1 = d1.forward(m0)
        ps(m1)
        a_m1 = tan.forward(m1)
        ps(a_m1)


        m2 = out.forward(a_m1)
        ps(m2)
        a_m2 = tan.forward(m2)
        ps(a_m2)
        loss = Mse.forward(a_m2, target)
        a, prediction = a_m2.view(-1).min(0)
        #print("n loss:", loss)
        #print(a_m2)
        if train_test != 1:
            e0 = Mse.backward(a_m2, target)  # M
            ps(e0)
            slope_out = tan.backward(a_m2)  # A
            ps(slope_out)
            c_e0 = e0 *slope_out
            ps(c_e0)
            e1 = out.backward(c_e0)  # D
            ps(e1)

            slope_hid = tan.backward(a_m1)  # A
            ps(slope_hid)
            c_e1 = e1 * slope_hid
            ps(c_e1)
            e2 = d1.backward(c_e1)  # D -useless-
            ps(e2)


            out.compute_gradient(a_m1, c_e0)
            out.apply_gradient(learning_rate)

            d1.compute_gradient(m0, c_e1)
            d1.apply_gradient(learning_rate)
        #w = a_m1.mm(c_e0)
        #b = torch.sum(c_e0)
        #out.apply_gradient(w, b, learning_rate)

        #w = torch.t(m0).mm(c_e1)
        #b = torch.sum(c_e1)  # sum bias??? on error? Dimension msimactch
        #d1.apply_gradient(w, b, learning_rate)


        predictions.append(prediction[0])

        total_loss += float(loss)
        message = str('Training loss %3f - iteration %i/%i, epoch %i/%i' % (loss, j, point_number, train_test, 2))
        log.info(message)
        #exit()
print(predictions)
print(train_targets)
final_tr_loss += total_loss / (j + 1)
train_accuracy = util.data.compute_accuracy(train_targets[:, 0], predictions)
final_tr_acc += train_accuracy
message = str('Average Training loss %3f - epoch %i/%i' % (total_loss / (j + 1), (0 + 1), 1))
log.info(message)
message = str('Average Training accuracy %3f - epoch %i/%i' % (train_accuracy, (0 + 1), 1))
log.info(message)
util.data.display_data_set(opt, train_examples, predictions,name="train_predictions", format='Normal')

exit()

d1 = Dense(2, 25)
h1 = Dense(25, 25)
h2 = Dense(25, 25)
h3 = Dense(25, 25)
out = Dense(25, 2)

tan = Tanh()
relu = ReLu()

Mse = MSE()

layers = [d1, relu, h1, relu, h2, tan, h3, tan, out, tan, Mse]
model = framework.modules.sequential.Sequential(layers=layers)
point_number = opt['point_number']

train_examples, train_targets = util.data.generate_toy_data()# util.data.generate_data(points_number=point_number)
test_examples, test_targets = util.data.generate_toy_data() #generate_data(points_number=point_number)

util.data.display_data_set(opt, train_examples, train_targets[:, 0])
epochs = 1
final_tr_loss = 0.0
final_te_loss = 0.0
final_tr_acc = 0.0
final_te_acc = 0.0
predictions = []
predictions_test = []
for i in range(epochs):
    total_loss = 0.0
    for j in range(0, point_number):
        target = train_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        train_data = train_examples[j].view(1, 2)
        loss, (_, prediction) = model.forward(train_data, target)
        predictions.append(prediction[0])
        model.backward(target, learning_rate=opt['lr'])
        total_loss+=float(loss)
        message = str('Training loss %3f - iteration %i/%i, epoch %i/%i' %(loss, j, point_number, i, epochs))
        log.info(message)
    final_tr_loss += total_loss/(j+1)
    train_accuracy = util.data.compute_accuracy(train_targets[:, 1], predictions)
    final_tr_acc += train_accuracy
    message = str('Average Training loss %3f - epoch %i/%i' % (total_loss/(j+1), (i+1), epochs))
    message = str('Average Training accuracy %3f - epoch %i/%i' % (train_accuracy, (i+1), epochs))

    log.info(message)
    total_loss = 0.0
    for j in range(0, point_number):
        target = test_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        test_data = test_examples[j].view(1, 2)
        loss, (_, prediction) = model.forward(train_data, target)
        predictions_test.append(prediction[0])
        total_loss += float(loss)
        message = str('Testing loss %3f - iteration %i/%i, epoch %i/%i' % (loss, j, point_number, i, epochs))
        log.info(message)
    test_accuracy = util.data.compute_accuracy(test_targets[:, 1], predictions_test)
    final_te_acc += test_accuracy
    message = str('Average Testing loss %3f - epoch %i/%i' % (total_loss / (j + 1), (i + 1), epochs))
    message = str('Average Testing accuracy %3f - epoch %i/%i' % (test_accuracy, (i + 1), epochs))
    log.info(message)
    final_te_loss += total_loss/(j+1)
print("-" * 60)
print("Average Training loss accross all the epochs: %3f" %(final_tr_loss/epochs))
print("Average Training accuracy accross all the epochs: %3f" %(final_tr_acc/epochs))
print("-" * 60)
print("Average Testing loss accross all the epochs: %3f" %(final_te_loss/epochs))
print("Average Testing accuracy accross all the epochs: %3f" %(final_te_acc/epochs))
util.data.display_data_set(opt, test_examples, predictions_test,name="test_predictions", format='Normal')
util.data.display_data_set(opt, train_examples, predictions,name="train_predictions", format='Normal')