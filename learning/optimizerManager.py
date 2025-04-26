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
    """add custom losses here"""
    loss = configuration.loss
    if loss == "mse":
        return torch.nn.MSELoss()
    elif loss == "crossentropy":
        # TODO add reduction?
        return torch.nn.CrossEntropyLoss
    else:
        raise NotImplementedError
