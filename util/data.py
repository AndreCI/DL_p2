import numpy as np
import math
import matplotlib.pyplot as plt


def generate_data(points_number=1000, disk_radius=1.0/(math.sqrt(2.0*math.pi))):
    '''
    Generate a data set of points sampled between [0, 1]² each with a label 0 if outside a disk of radius disk_radius,
    and 1 if outside
    :param points_number: the number of points to sample
    :param disk_radius: the disk radius
    :return: a dataset
    '''
    data_set = np.random.uniform(0, 1, (points_number, 4)) #TODO: not nice
    for ex in data_set:
        dist = math.sqrt(ex[0]**2 + ex[1]**2)
        if dist>disk_radius:
            ex[2] = 1
            ex[3] = 0
        else:
            ex[2] = 0
            ex[3] = 1
    return data_set

def generate_toy_data(points_number=1000):
    data_set = np.random.uniform(0, 1, (points_number, 4))
    for ex in data_set:
        dist = ex[0] + ex[1]
        if dist>0.9:
            ex[2] = 1
            ex[3] = 0
        else:
            ex[2] = 0
            ex[3] = 1
    return data_set

def display_data_set(examples, targets):
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
    plt.savefig("dataset.jpg")