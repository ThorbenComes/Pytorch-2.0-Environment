import torch.utils.data

from models_layers import modelManager
from learning import optimizerManager
from learning import learningManager
import wandb
from omegaconf import OmegaConf
import flatten_dict
import utils.logs.logWriter
from experiment_data import dataLoader


class Experiment:
    def __init__(self):
        self.configuration = wandb.config
        print(self.configuration)
        # self.data_dict = dataLoader.load(configuration)

    def _init(self):

        wandb_run = self._wandb_init()
        config = OmegaConf.create(flatten_dict.unflatten_dict(wandb.config))
        self.cfg = config

        self.model = modelManager.model(config)
        self.optimizer = optimizerManager.optimizer(config, self.model.parameters())
        self.learning_algorithm = learningManager.algorithm(config)
        self.train_dataset = torch.utils.data.TensorDataset(self.data_dict["train_obs"], self.data_dict["train_targets"])
        self.eval_dataset = torch.utils.data.TensorDataset(self.data_dict["test_obs"], self.data_dict["test_targets"])
        self.logWriter = utils.logs.logWriter.logWriter(config, wandb_run)

    def train(self):
        print("starting training")
        pass

    def eval(self):
        print("starting evaluation")
        pass

