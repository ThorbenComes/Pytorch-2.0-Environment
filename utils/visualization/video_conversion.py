import wandb
import numpy as np
import torch


def get_mp4(tensor, fps=1):
    """
    :param tensor: (frame x channel x height x width) or (batch x frame x channel x height x width)
    The tensor is expected to have either 1 or 3 channels.
    The tensor values should be within [0, 1]
    :return: wandb video in mp4 format
    """
    # tensor is converted to uint 8 space and must be scaled beforehand
    tensor *= 255
    if tensor.shape[-3] == 1:
        tensor = torch.cat((tensor, tensor, tensor), dim=-3)
    return wandb.Video(tensor.cpu().numpy(), format="mp4", fps=fps)
