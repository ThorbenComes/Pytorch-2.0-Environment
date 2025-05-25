from utils.functions import nn

# TODO: gaussian nll, has additional argument of variance.
# possible solution: lambda function with constant variance -> 2 arg loss.


def get_loss(loss_name):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mse_sum":
        return nn.MSELoss(reduction="sum")
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "nll":
        return nn.NLLLoss()
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "kl_divergence":
        return nn.KLDivLoss()
    else:
        raise ValueError("Unknown loss name: {}".format(loss_name))
