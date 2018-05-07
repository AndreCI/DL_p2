import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch import FloatTensor, LongTensor
import os
import math
import numpy as np


def generate_data(points_number=1000, disk_radius=1.0/(math.sqrt(2.0*math.pi))):
    '''
    Generate a data set of points sampled between [0, 1]² each with a label 0 if outside a disk of radius disk_radius,
    and 1 if outside
    :param points_number: the number of points to sample
    :param disk_radius: the disk radius
    :return: a tuple containing the examples and their associated labels
    '''
    examples = FloatTensor(points_number, 2).uniform_(-0.5, 0.5)
    targets = LongTensor(points_number, 2).zero_()
    for i,ex in enumerate(examples):
        dist = math.sqrt(ex[0]**2 + ex[1]**2)
        if dist>disk_radius: #or ex[0] < 0:
            targets[i,0] = 1
        else:
            targets[i,1] = 1
        ex[0] += 0.5
        ex[1] += 0.5
    return examples, targets

def generate_toy_data(points_number=1000, dist=1.0/(math.sqrt(1.0*math.pi))):
    '''
    Generate a data set of points sampled between [0, 1]² each with a label 0 if x + y < dist
    and 1 if outside
    :param points_number: the number of points to sample
    :param dist: the value to cut the plane with
    :return: a tuple containing the examples and their associated labels
    '''
    examples = FloatTensor(points_number, 2).uniform_(0, 1)
    targets = LongTensor(points_number, 2).zero_()
    for i, ex in enumerate(examples):
        current_dist = math.sqrt(ex[0]**2 + ex[1]**2)
        if current_dist > dist:
            targets[i, 0] = 1
        else:
            targets[i, 1] = 1
    return examples, targets

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

def compute_accuracy(targets, predictions):
    return sum((targets.numpy() == predictions))/targets.size()[0]