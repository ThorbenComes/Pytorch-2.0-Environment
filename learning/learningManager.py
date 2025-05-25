import learning.forward_pass
import torch.nn as nn
import utils.metrics.losses


def algorithm(configuration):
    """returns learning algorithm according to configuration"""
    return simpleLearning(configuration)


class simpleLearning:
    def __init__(self, config):
        self.loss = utils.metrics.losses.get_loss(config.learning.loss)

    def learn(self, model, optimizer, dataset):
        """implements one iteration of the learning algorithm"""
        current_loss = learning.forward_pass.standard_epoch_optimizer(model, dataset, self.loss, optimizer)
        return current_loss


