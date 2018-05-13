from torch import FloatTensor, LongTensor
import math


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

def compute_accuracy(targets, predictions):
    '''
    Comput the accuracy between the expected targets and the predictions
    :param targets: the targets
    :param predictions: the predictions of the model
    :return: a value between 0 and 1 representing the number of correct prediction
    '''
    return sum((targets.numpy() == predictions))/targets.size()[0]