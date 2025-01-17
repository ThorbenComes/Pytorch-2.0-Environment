import sys

import Tests.test_data_loading

""" These shenanigans will hopefully no longer be necessary
sys.path.append('.')
currentwd = os.getcwd()
new_path = currentwd.split("outputs", 1)[0]
if new_path[-1] in ["\\", "/"]:
    new_path = new_path[:-1]
sys.path.append(new_path)"""
from omegaconf import DictConfig, OmegaConf
import hydra
import os



import glob
import torch
import pickle
import wandb

from hydra.utils import get_original_cwd

####################################New imports
from utils import wandb_utils
from utils import hydra_utils
from experiment_data import timeseriesExperiment
from Tests import test_data_loading
import experiment_data.experiment_manager
from functools import partial
import copy


@hydra.main(config_path='configurations', config_name="default_config")
def my_app(cfg) -> OmegaConf:
    """Main method


    """
    debug_all = False
    debug = debug_all or False

    if debug_all:
        Tests.test_data_loading.test_load_kitti()

    cfg_omega = OmegaConf.to_container(cfg)
    cfg_s = wandb_utils.create_wandb_sweep_config(copy.deepcopy(cfg_omega))

    # avoid 30s wandb termination
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    # TODO write config, make configurable

    # load data dictionary
    # data_dict = dataLoader.load(cfg_s)
    # initialize experiment
    # TODO: make experiment type configurable

    if debug:
        hydra_utils.prettyPrint(cfg_s, message="------------------Sweep Configuration-----------------")

    sweep_id = wandb.sweep(cfg_s, project=cfg.wandb.project_name)

    if debug:
        print("launching agent")
    # wandb.agent(sweep_id, experiment.init_train_eval_model, count=wandb_utils.calculate_number_of_runs(cfg_s))
    partial_function = partial(experiment_data.experiment_manager.conduct_experiment,
                               mode = cfg_omega["wandb"]["logging"],
                               project_name = cfg_omega["wandb"]["project_name"],
                               name = cfg_omega["wandb"]["name"])
    wandb.agent(sweep_id, partial_function, count=wandb_utils.calculate_number_of_runs(cfg_s))


def main():
    my_app()


## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()
