import torch.nn as nn


def identity(x):
    return x


class ImputationModule(nn.Module):
    """
    Wrapper for nn Modules to include imputation
    """

    def __init__(self, model):
        super().__init__()
        self.mask = identity
        self.model = model

    def set_imputation(self, function):
        self.mask = function

    def reset_imputation(self):
        self.mask = identity

    def forward(self, x):
        x = self.mask(x)
        y = self.model(x)
        return y