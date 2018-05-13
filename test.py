from framework.modules.trainable_modules.dense_layer import DenseLayer as Dense
from framework.modules.criterion_modules.mse_layer import MSELayer as MSE
from framework.modules.activation_modules.relu_layer import ReLuLayer as ReLu
from framework.modules.activation_modules.tanh_layer import TanhLayer as Tanh
from framework.modules.activation_modules.sigmoid_module import SigmoidLayer as Sigmoid
from framework.optimizers.sgd_optimizer import SGD_optimizer
from framework.initializers.he_initializer import HeInitializer
from framework.initializers.gaussian_initializer import GaussianInitializer
from framework.initializers.uniform_initializer import UniformInitializer
from framework.initializers.xavier_initializer import XavierInitializer
from framework.modules.sequential import Sequential
import framework.modules.sequential
import util.data_generation
from torch import FloatTensor


#Model info
verbose = 'low'
hidden_units = 25
point_number = 1000
epochs = 50
learning_rate = 0.01
momentum = 0.0
initializer = HeInitializer()


#Creation of the different dense layers
d1 = Dense(2, hidden_units, use_bias=True, initializer=initializer)
h1 = Dense(hidden_units, hidden_units, use_bias=True, initializer=initializer)
h2 = Dense(hidden_units, hidden_units, use_bias=True, initializer=initializer)
h3 = Dense(hidden_units, hidden_units, use_bias=True, initializer=initializer)
out = Dense(hidden_units, 2)

#creation of the activations layers
tan = Tanh()
relu = ReLu()
sig = Sigmoid()

#creation of the criterion layer
Mse = MSE()

#model creation
layers = [d1, tan, h1, tan, h2, tan, h3, tan, out, tan, Mse]
model = framework.modules.sequential.Sequential(layers=layers)

#optimizer to train the model
optimizer = SGD_optimizer(model, learning_rate, momentum)

#generation of the train and test dataset
train_examples, train_targets = util.data_generation.generate_data(points_number=point_number)
test_examples, test_targets =util.data_generation.generate_data(points_number=point_number)

final_tr_loss = []
final_te_loss = []
final_tr_acc = []
final_te_acc = []

for i in range(epochs):
    predictions = []
    predictions_test = []
    total_loss = 0.0
    for j in range(0, point_number): #With SGD, we iterate over the dataset
        target = train_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        train_data = train_examples[j].view(1, 2)
        loss, (_, prediction) = model.forward(train_data, target) #forward pass
        predictions.append(prediction[0])
        optimizer.step(target) #backward pass
        total_loss+=float(loss)

        if verbose == 'high': #logging of loss at each example is verbose is high
            message = str('Training loss %.3f - iteration %i/%i, epoch %i/%i' %(loss, j, point_number, i, epochs))
            print(message)
    #End of training epoch, now we compute accuracy and overall loss and display it
    final_tr_loss.append(total_loss/(point_number))
    train_accuracy = util.data_generation.compute_accuracy(train_targets[:, 0], predictions)
    final_tr_acc.append(train_accuracy)
    message = str('Training loss %.3f - epoch %i/%i' % (total_loss/(point_number), (i+1), epochs))
    print(message)
    message = str('Training accuracy %.3f - epoch %i/%i' % ((train_accuracy), (i+1), epochs))
    print(message)
    total_loss = 0.0
    #Testing the model after each training epoch
    for j in range(0, point_number):
        target = test_targets[j]
        target = target.view(1, 2).type(FloatTensor)
        test_data = test_examples[j].view(1, 2)
        loss, (_, prediction) = model.forward(test_data, target) #forward pass, but no backward pass
        predictions_test.append(prediction[0])
        total_loss += float(loss)
        if verbose == 'high':#logging of loss at each example is verbose is high
            message = str('Testing loss %3.f - iteration %i/%i, epoch %i/%i' % (loss, j, point_number, i, epochs))
            print(message)
    #End of testing epoch, now we compute accuracy, overall loss and display
    final_te_loss.append(total_loss/(point_number))
    test_accuracy = util.data_generation.compute_accuracy(test_targets[:, 0], predictions_test)
    final_te_acc.append(test_accuracy)
    message = str('Testing loss %.3f - epoch %i/%i' % ((total_loss) / (point_number), (i + 1), epochs))
    print(message)
    message = str('Testing accuracy %.3f - epoch %i/%i' % ((test_accuracy), (i + 1), epochs))
    print(message)
    print("-" * 60)
#End of program. Display overall accuracy and overall loss.
print("*" * 60)
print("Average Training loss accross all the epochs: %3f" %(sum(final_tr_loss)/epochs))
print("Average Training accuracy accross all the epochs: %3f" %(sum(final_tr_acc)/epochs))
print("*" * 60)
print("Average Testing loss accross all the epochs: %3f" %(sum(final_te_loss)/epochs))
print("Average Testing accuracy accross all the epochs: %3f" %(sum(final_te_acc)/epochs))
