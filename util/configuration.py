import os
import argparse
import sys
import logging
import json
from framework.modules.sequential import Sequential

def get_args(parser):
    '''
    Setup different infos
    :param parser: the user inputs
    :return: the list of the infos, with default values if not specified by the user
    '''
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    #Directory arguments
    parser.add_argument('--fig_dir', help="directory to save different figures.", default=root_dir + "/figs/", type=str)
    parser.add_argument('--log_root', help="directory to save the logging journal", default=root_dir + "/logs/", type=str)
    parser.add_argument('--save_dir', help="directory to save the different models", default=root_dir + "/save/", type=str)

    #Model and data arguments
    parser.add_argument('--hidden_units', help="Number of hidden units to use.", default=25, type=int)
    parser.add_argument('--epoch_number', help="Number of epoch to train.", default=50, type=int)
    parser.add_argument('--lr', help="Learning rate to train the models.", default=0.01, type=float)
    parser.add_argument('--momentum', help="Value for the momentum parameter in SGD", default=0.0, type=float)
    parser.add_argument('--point_number', help="Number of points to generate.", default=1000, type=int)
    parser.add_argument('--load_best_model', help="If True, the model with the most testing accuracy from the save_dir will be loaded and trained.",
                        default=False, type=bool)
    parser.add_argument('--save_best_model', help="If True, each model that beat the previous one will be saved in save_dir.", default=False, type=bool)

    parser.add_argument('--verbose', help="How much information will the log give. Options are 'high' or 'low'.", default='low', type=str)

    return vars(parser.parse_args())

def setup_log(opt):
    '''
    Setup a log.
    :param opt: the different options
    :return: the log, ready to use.
    '''
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    if not os.path.exists(opt['log_root']): os.mkdir(opt['log_root'])
    log_dir = os.path.join(opt['log_root'], str('output.log'))
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    log.info('[Program starts.]')
    important_infos = []
    for key in opt:
        if type(opt[key]) is bool:
            if opt[key]:
                important_infos.append({key: opt[key]})
        elif type(opt[key]) is int:
            if opt[key] > 0:
                important_infos.append({key: opt[key]})
        elif type(opt[key]) is str and 'dir' not in key:
            important_infos.append({key: opt[key]})
    log.info('[Arg used:]' + str(important_infos))
    return log

def load_most_successful_model(save_dir):
    '''
    Load the best model based on test accuracy
    :param save_dir: the path in where the method will look
    :return: The best model found, if any.
    '''
    best_acc = 0.0
    best_file = None
    for f in os.listdir(save_dir):
        if not os.path.isdir(f):
            file = os.path.join(save_dir, f)
            with open(file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if data['test_accuracy'] > best_acc:
                    best_acc = data['test_accuracy']
                    best_file = f
    return Sequential.load_model(best_file[:-5], save_dir)