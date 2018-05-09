from framework.modules.trainable_modules.dense_layer import DenseLayer as Dense
from framework.modules.criterion_modules.mse_layer import MSELayer as MSE
from framework.modules.activation_modules.relu_layer import ReLuLayer as ReLu
from framework.modules.activation_modules.tanh_layer import TanhLayer as Tanh
from framework.modules.activation_modules.sigmoid_module import SigmoidLayer as Sigmoid
from framework.optimizers.sgd_optimizer import SGD_optimizer
from framework.modules.sequential import Sequential
import framework.modules.sequential
import util.data
from torch import FloatTensor, LongTensor
from util.configuration import get_args, setup_log, load_most_successful_model
import argparse

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

d1 = Dense(2, 25)
h1 = Dense(25, 25)
h2 = Dense(25, 25)
h3 = Dense(25, 25)
out = Dense(25, 2)

tan = Tanh()
relu = ReLu()
sig = Sigmoid()

Mse = MSE()


if opt['load_best_model']:
    model = load_most_successful_model(opt['save_dir'])
else:
    layers = [d1, tan, h1, tan, h2, tan, h3, tan, out, tan, Mse]
    model = framework.modules.sequential.Sequential(layers=layers)

optimizer = SGD_optimizer(model, opt['lr'], opt['momentum'])

point_number = opt['point_number']

train_examples, train_targets = util.data.generate_data(points_number=point_number) #util.data.generate_toy_data(opt['point_number']) #
test_examples, test_targets =util.data.generate_data(points_number=point_number) #util.data.generate_toy_data(opt['point_number']) #

util.data.display_data_set(opt, train_examples, train_targets[:, 0], name='train_dataset')
util.data.display_data_set(opt, test_examples, test_targets[:, 0], name='test_dataset')
epochs = opt['epoch_number']
final_tr_loss = 0.0
final_te_loss = 0.0
final_tr_acc = 0.0
final_te_acc = 0.0
for i in range(epochs):
    predictions = []
    predictions_test = []
    total_loss = 0.0
    for j in range(0, point_number):
        target = train_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        train_data = train_examples[j].view(1, 2)
        loss, (jack, prediction) = model.forward(train_data, target)
        predictions.append(prediction[0])
        optimizer.step(target)
        #model.backward(target, learning_rate=opt['lr'])
        total_loss+=float(loss)
        if opt['verbose'] == 'high':
            message = str('Training loss %3f - iteration %i/%i, epoch %i/%i' %(loss, j, point_number, i, epochs))
            log.info(message)
    final_tr_loss += total_loss/(j+1)
    train_accuracy = util.data.compute_accuracy(train_targets[:, 0], predictions)
    final_tr_acc += train_accuracy
    message = str('Average Training loss %3f - epoch %i/%i' % (total_loss/(j+1), (i+1), epochs))
    message = str('Average Training accuracy %3f - epoch %i/%i' % (train_accuracy, (i+1), epochs))

    log.info(message)
    total_loss = 0.0
    for j in range(0, point_number):
        target = test_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        test_data = test_examples[j].view(1, 2)
        loss, (_, prediction) = model.forward(test_data, target)
        predictions_test.append(prediction[0])
        total_loss += float(loss)
        if opt['verbose'] == 'high':
            message = str('Testing loss %3f - iteration %i/%i, epoch %i/%i' % (loss, j, point_number, i, epochs))
            log.info(message)
    test_accuracy = util.data.compute_accuracy(test_targets[:, 0], predictions_test)
    final_te_acc += test_accuracy
    message = str('Average Testing loss %3f - epoch %i/%i' % (total_loss / (j + 1), (i + 1), epochs))
    message = str('Average Testing accuracy %3f - epoch %i/%i' % (test_accuracy, (i + 1), epochs))
    log.info(message)
    if test_accuracy < 0.6:
        model.reset()
        log.info(str('A reset has occured.'))
    else:
        name = str('model_%i_%.3fAcc' %(i, test_accuracy))
        model.save_model(name, opt['save_dir'], test_acc=test_accuracy)
    final_te_loss += total_loss/(j+1)
    util.data.display_data_set(opt, test_examples, predictions_test,name="test_predictions", format='Normal')
    util.data.display_data_set(opt, train_examples, predictions,name="train_predictions", format='Normal')
print("-" * 60)
print("Average Training loss accross all the epochs: %3f" %(final_tr_loss/epochs))
print("Average Training accuracy accross all the epochs: %3f" %(final_tr_acc/epochs))
print("-" * 60)
print("Average Testing loss accross all the epochs: %3f" %(final_te_loss/epochs))
print("Average Testing accuracy accross all the epochs: %3f" %(final_te_acc/epochs))