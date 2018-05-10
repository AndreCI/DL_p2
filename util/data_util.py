import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

from torch import FloatTensor, LongTensor
import os
import math
import numpy as np


def display_data_set(opt, examples, targets, name='dataset', format='torch'):
    '''
    Display the dataset. Save the fig under name
    :param examples: The list of examples
    :param targets: The list of their targets
    '''
    examples = examples.numpy()
    if format == 'torch':
        targets = targets.numpy()
    elif format != 'numpy':
        targets = np.array(targets)
    f = plt.figure(figsize=(20, 20))
    ax = f.add_subplot(111)
    condition = targets == 1
    ncon = targets == 0
    true_x = np.extract(condition, examples[:, 0])
    true_y = np.extract(condition, examples[:, 1])
    false_x = np.extract(ncon, examples[:, 0])
    false_y = np.extract(ncon, examples[:, 1])
    ax.plot(true_x, true_y, 'ro', false_x, false_y, 'bo')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    circle = plt.Circle((0.5, 0.5), 1.0/(math.sqrt(2.0*math.pi)), color='g', fill=False)
    ax.add_artist(circle)
    sname = str(name + '.jpg')
    if not os.path.exists(opt['fig_dir']) : os.mkdir(opt['fig_dir'])
    sname = os.path.join(opt['fig_dir'], sname)
    red_patch = mpatches.Patch(color='red', label='Label 1')
    blue_patch = mpatches.Patch(color='blue', label='Label 0')
    plt.legend(handles=[red_patch, blue_patch])
    plt.savefig(sname)
    plt.close()


def display_losses(train_loss, test_loss, model_type, opt, running_mean_param=1):
    if running_mean_param > len(train_loss) or running_mean_param > len(test_loss):
        running_mean_param = 1
    font = {'family': 'sherif',
            'size': 18}

    matplotlib.rc('font', **font)

    fig_width = 10
    golden_mean = (math.sqrt(5)-1.0)/2.0
    fig_height = fig_width * golden_mean
    train_loss = running_mean(train_loss, N=running_mean_param)
    test_loss = running_mean(test_loss, N=running_mean_param)
    plt.figure(figsize=(fig_width, fig_height))
    title = str('Evolution of train and test loss')
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Testing loss'])
    name = str('%s_loss.png' % (model_type))
    loc = os.path.join(opt['fig_dir'], name)
    plt.savefig(loc)

def display_accuracy(train_accuracy, test_accuracy, model_type, opt, running_mean_param=1):
    if running_mean_param > len(train_accuracy) or running_mean_param > len(test_accuracy):
        running_mean_param = 1
    train_accuracy = running_mean(train_accuracy, N=running_mean_param)
    test_accuracy = running_mean(test_accuracy, N=running_mean_param)
    font = {'family': 'sherif',
            'size': 18}

    matplotlib.rc('font', **font)
    fig_width = 10
    golden_mean = (math.sqrt(5)-1.0)/2.0
    fig_height = fig_width * golden_mean

    plt.figure(figsize=(fig_width, fig_height))

    title = str('Evolution of train and test accuracy')
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.legend(['Training accuracy', 'Testing accuracy'])
    name = str('%s_accuracy.png' % (model_type))
    loc = os.path.join(opt['fig_dir'], name)
    plt.savefig(loc)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)