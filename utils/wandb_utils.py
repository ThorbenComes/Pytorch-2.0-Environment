import flatten_dict
import wandb
"""Class offering utilities for weights and biases"""


def create_wandb_sweep_config(configuration):
    # TODO: switch case for different searches
    configuration = {"parameters": {"model": {"parameters": translateToWandbParams(configuration)}}, "method": "grid"}
    return configuration


def translateToWandbParams(config_dict: dict):
    """
    Adapts a configuration to be readable by Wandb
    The parameters keyword is added on all parameters, as well as between every layer of nested parameters
    The value keyword is added before any value not specified in the parameters part of the config
    The values keyword is added before any values specified in the parameters part of the config
    The parameters specified in the parameters part of the config overwrite the
    """
    parameters = config_dict["parameters"]
    parameters = add_parameters_keyword(parameters)

    params_dict_flat = flatten_dict.flatten(parameters, reducer="dot")

    # add value and values keywords
    for key, value in params_dict_flat.items():
        if isinstance(value, list):
            params_dict_flat[key] = {"values": value}
        else:
            params_dict_flat[key] = {"value": value}

    del config_dict["parameters"]

    config_dict = add_parameters_keyword(config_dict)
    dict_flat = flatten_dict.flatten(config_dict, reducer="dot")

    for key, value in dict_flat.items():
        dict_flat[key] = {"value": value}

    # update all values with those from parameters part of config
    # allows for overwriting of all values
    dict_flat.update(params_dict_flat)
    config = flatten_dict.unflatten(dict_flat, splitter="dot")

    return config


def add_parameters_keyword(config: dict):
    """
    This function adds the parameters keyword in between each layer of dictionaries.
    This is necessary since wandb does not support nested parameters without this.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = {"parameters": add_parameters_keyword(value)}
    return config


def calculate_number_of_runs(config_dict: dict):
    """
    Calculates number of runs to make the agent stop once all combinations have been tried
    """
    dict_flat = flatten_dict.flatten(config_dict, reducer="dot")
    runs = 1
    for key, value in dict_flat.items():
        if ".values" in key:
            runs *= len(value)
    return runs
