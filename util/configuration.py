import os
import argparse
import sys
import logging


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

    #Model and data arguments
    parser.add_argument('--epoch_number', help="Number of epoch to train.", default=100, type=int)
    parser.add_argument('--lr', help="Learning rate to train the models.", default=0.001, type=float)
    parser.add_argument('--point_number', help="Number of points to generate.", default=10000, type=int)

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