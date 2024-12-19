from models_layers import modelManager
from experiment_data import optimizerManager
import wandb
from omegaconf import OmegaConf
from credentials import wandb_api_key


class Experiment:
    def __init__(self, configuration, data_dict):

        self.data_dict = data_dict

    def _wandb_init(self, wandb_login = False):
        ## Convert Omega Config to Wandb Config
        config_dict = OmegaConf.to_container(self.cfg)
        try:
            exp_name = self.model_cfg.wandb.exp_name + self.model_cfg.parameters.learn.name
        except:
            exp_name = self.model_cfg.wandb.exp_name + self.learn_cfg.name
        if self.model_cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"
        ## Initializing wandb object and sweep object
        # if self.model_cfg.wandb.log:
        # wandb.login(key=wandb_api_key, relogin=True)
        # self.wandb_config = wandb.config
        if wandb_login:
            wandb.login(key=wandb_api_key, relogin=True)
        wandb_run = wandb.init(config=config_dict, project=self.model_cfg.wandb.project_name, name=exp_name,
                               mode=mode)  # wandb object has a set of configs associated with it as well
        return wandb_run

    def init_train_eval_model(self):
        wandb_run = self._wandb_init()
        pass
