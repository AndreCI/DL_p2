import numpy as np
import math


def generate_data(points_number=1000, disk_radius=1.0/(math.sqrt(2.0*math.pi))):
    '''
    Generate a data set of points sampled between [0, 1]² each with a label 0 if outside a disk of radius disk_radius,
    and 1 if outside
    :param points_number: the number of points to sample
    :param disk_radius: the disk radius
    :return: a dataset
    '''
    data_set = np.random.uniform(0, 1, (points_number, 3)) #TODO: not nice
    for ex in data_set:
        dist = np.linalg.norm(ex[0] - ex[1])
        if dist>disk_radius:
            ex[2] = 1
        else:
            ex[2] = 0
    return data_set