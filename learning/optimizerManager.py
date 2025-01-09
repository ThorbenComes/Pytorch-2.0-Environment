import torch


"""configures and chooses optimizer and loss function based on configuration"""


def optimizer(configuration, parameters):
    lr = configuration.lr
    # TODO: get config right
    return torch.optim.Adam(parameters, lr = lr)


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
