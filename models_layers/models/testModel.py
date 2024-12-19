import torch

nn = torch.nn

"""simple model for testing experiments"""


class testModel(nn.Module):

    def __init__(self, config):
        self.layers = nn.Sequential()  # TODO: add simple model
        pass

    def forward(self, x, flags=None):
        return self.layers(x)
