import os
import sys

import torch
from hydra.utils import get_original_cwd
import utils.hydra_utils
"""Class for selecting and loading data"""


def load(configuration):
    """loads dataset as according to given configuration"""
    # utils.hydra_utils.prettyPrint(configuration, message="loading")
    location = configuration.dataset.directory + configuration.dataset.filename
    return load_file(location)
    # TODO: get value of dataset

    # TODO: switch case for different datasets
    # pass


def load_file(filename):
    """
    write a function to load the data and return the train and test dataset
    :return: data dictionary containing ambiguous data
    """
    if not os.path.exists(get_original_cwd() + filename):
        print("..........Data Not Found...........stopping")
        sys.exit("This dataset cannot be downloaded")
    else:
        print("..........Data Found...........Loading from local")
    with open(get_original_cwd() + filename, 'rb') as f:
        data_dict = torch.load(f)
    return data_dict

