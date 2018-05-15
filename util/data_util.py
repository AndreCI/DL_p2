import math
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def display_data_set(opt, examples, targets, name='dataset'):
    """
    Display the dataset. Save the fig under name
    :param opt: The option parameters, which contains info such as save_dir
    :param name: the name of the image
    :param examples: The list of examples
    :param targets: The list of their targets
    """
    examples = examples.numpy()
    f = plt.figure(figsize=(20, 20))
    ax = f.add_subplot(111)
    condition = targets == 1
    if type(condition) is bool:
        condition = [condition for _ in targets]
    true_x = []
    false_x = []
    for i, temp_x in enumerate(examples[:, 0]):
        if condition[i]:
            true_x.append(temp_x)
        else:
            false_x.append(temp_x)
    true_y = []
    false_y = []
    for i, temp_y in enumerate(examples[:, 1]):
        if condition[i]:
            true_y.append(temp_y)
        else:
            false_y.append(temp_y)
    ax.plot(true_x, true_y, 'ro', false_x, false_y, 'bo')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    circle = plt.Circle((0.5, 0.5), 1.0 / (math.sqrt(2.0 * math.pi)), color='g', fill=False)
    ax.add_artist(circle)
    sname = str(name + '.jpg')
    if not os.path.exists(opt['fig_dir']): os.mkdir(opt['fig_dir'])
    sname = os.path.join(opt['fig_dir'], sname)
    red_patch = mpatches.Patch(color='red', label='Label 1')
    blue_patch = mpatches.Patch(color='blue', label='Label 0')
    plt.legend(handles=[red_patch, blue_patch])
    plt.savefig(sname)
    plt.close()


def display_losses(train_loss, test_loss, model_type, opt):
    font = {'family': 'sherif',
            'size': 18}

    matplotlib.rc('font', **font)

    fig_width = 10
    golden_mean = (math.sqrt(5) - 1.0) / 2.0
    fig_height = fig_width * golden_mean
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
    name = str('%s_loss.png' % model_type)
    loc = os.path.join(opt['fig_dir'], name)
    plt.savefig(loc)


def display_accuracy(train_accuracy, test_accuracy, model_type, opt):
    font = {'family': 'sherif',
            'size': 18}

    matplotlib.rc('font', **font)
    fig_width = 10
    golden_mean = (math.sqrt(5) - 1.0) / 2.0
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
    name = str('%s_accuracy.png' % model_type)
    loc = os.path.join(opt['fig_dir'], name)
    plt.savefig(loc)
