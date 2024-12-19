import os
import torch
from hydra.utils import get_original_cwd
"""Class for selecting and loading data"""


def load(configuration):
    """loads dataset as according to given configuration"""
    # TODO: get value of dataset

    # TODO: switch case for different datasets
    pass


def load_file(filename):
    """
    write a function to load the data and return the train and test data
    :return: data dictionary containing ambiguous data
    """
    if not os.path.exists(get_original_cwd() + filename):
        print("..........Data Not Found...........stopping")
        assert (False, "This dataset cannot be downloaded")
    else:
        print("..........Data Found...........Loading from local")
    with open(get_original_cwd() + filename) as f:
        data_dict = torch.load(f)
    return data_dict

