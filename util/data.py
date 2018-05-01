import numpy as np
import math
import matplotlib.pyplot as plt
from torch import FloatTensor, LongTensor


def generate_data(points_number=1000, disk_radius=1.0/(math.sqrt(2.0*math.pi))):
    '''
    Generate a data set of points sampled between [0, 1]² each with a label 0 if outside a disk of radius disk_radius,
    and 1 if outside
    :param points_number: the number of points to sample
    :param disk_radius: the disk radius
    :return: a tuple containing the examples and their associated labels
    '''
    examples = FloatTensor(points_number, 2).uniform_(0, 1)
    targets = LongTensor(points_number, 2).zero_()
    for i,ex in enumerate(examples):
        dist = math.sqrt(ex[0]**2 + ex[1]**2)
        if dist>disk_radius:
            targets[i,0] = 1
        else:
            targets[i,1] = 1
    return examples, targets

def generate_toy_data(points_number=1000, dist=0.5):
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
        current_dist = ex[0] + ex[1]
        if current_dist > dist:
            targets[i, 0] = 1
        else:
            targets[i, 1] = 1
    return examples, targets

def display_data_set(examples, targets, name='dataset'):
    '''
    Display the dataset. Save the fig under name
    :param examples: The list of examples
    :param targets: The list of their targets
    '''
    examples = examples.numpy()
    targets = targets.numpy()
    f = plt.figure(figsize=(30, 30))
    ax = f.add_subplot(111)
    condition = targets == 1
    ncon = targets == 0
    true_x = np.extract(condition, examples[:, 0])
    true_y = np.extract(condition, examples[:, 1])
    false_x = np.extract(ncon, examples[:, 0])
    false_y = np.extract(ncon, examples[:, 1])
    ax.plot(true_x, true_y, 'ro', false_x, false_y, 'bo')
    f.show()
    sname = str(name + '.jpg')
    plt.savefig(sname)