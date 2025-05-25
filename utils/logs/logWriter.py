import wandb
import learning.forward_pass
import utils.metrics.losses
import utils.visualization.video_conversion
import torch


def get_losses_to_log(config):
    loss_dict = {}
    for loss_name in config.logging.test_losses:
        updated_dict = {loss_name: utils.metrics.losses.get_loss(loss_name)}
        loss_dict.update(updated_dict)
    return loss_dict


class logWriter:
    def __init__(self, config, wandb_run):
        """

        :param config:
        :param wandb_run:
        :param losses_to_log: dictionary of the form {loss_name: loss_function}
        """
        self.run = wandb_run
        self.losses_to_log = get_losses_to_log(config)
        self.log_vid = config.logging.log_video

    def log(self, model, test_dataset, train_loss):
        current_dict = {"train_loss": train_loss}
        current_dict.update(self.log_test_losses(model, test_dataset))
        self.run.log(current_dict)

    def log_test_losses(self, model, test_dataloader):
        """
        iterate once, log all losses

        If items(), keys(), values(), iteritems(), iterkeys(), and itervalues() are called with no intervening
        modifications to the dictionary, the lists will directly correspond.

        :param model:
        :param test_dataloader:
        :return:
        """
        current_dict = {}
        loss_keys = list(self.losses_to_log.keys())
        loss_functions = list(self.losses_to_log.values())
        losses = learning.forward_pass.evaluate_loss_functions(model, test_dataloader, loss_functions)

        for j in range(len(loss_keys)):
            current_dict.update({loss_keys[j]: losses[j]})

        return current_dict

    def log_video(self, model, test_dataset):
        with torch.no_grad():
            for input, target in test_dataset:
                result = model(input)[0]
                break
        return {"video": utils.visualization.video_conversion.get_mp4(result)}

    def log_evaluation(self, model, test_dataset):
        if self.log_vid:
            self.run.log(self.log_video(model, test_dataset))
