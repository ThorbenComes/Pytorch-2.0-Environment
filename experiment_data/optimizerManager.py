import torch


"""configures and chooses optimizer based on configuration"""


def optimizer(configuration, parameters):
    lr = configuration.lr
    # TODO: get config right
    return torch.optim.Adam(parameters, lr = lr)
