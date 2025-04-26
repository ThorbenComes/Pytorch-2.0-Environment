import torch


"""configures and chooses optimizer and loss function based on configuration"""


def optimizer(configuration, parameters):
    lr = configuration.learning.lr
    optimizer_type = configuration.learning.type
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif optimizer_type.type == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr)
    elif optimizer_type.type == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr)
    else:
        raise NotImplementedError("Optimizer {} not implemented".format(optimizer_type))
    return optimizer


def loss_function(configuration):
    """
    add custom losses here
    the loss_args can be used to specify loss function arguments as a dictionary of keyword arguments
    """
    loss = configuration.loss
    try:
        args = configuration.loss_args
    except AttributeError:
        args = {}
    if loss == "mse":
        return torch.nn.MSELoss(**args)
    elif loss == "crossentropy":
        return torch.nn.CrossEntropyLoss(**args)
    else:
        raise NotImplementedError("Loss {} not implemented".format(loss))
