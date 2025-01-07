"""does a forward pass through a model, with and without optimization"""
import torch


def standard_frame(model, dataLoader, loss_function, optimizer, function=None):
    """
    does a standard evaluation epoch
    :param function: function to be executed to implement optimization
    :param model: model to train
    :param dataLoader: torch dataloader with sources and targets
    :param optimizer: torch optimizer
    :param loss_function: custom loss function
    :return: cumulative loss of all batches
    """

    running_loss = 0.

    func = function

    with torch.no_grad:
        for i, data in enumerate(dataLoader):

            source, target = data

            output = model(source)

            loss = loss_function(output, target)

            running_loss += func(optimizer, loss)

    return running_loss


def optimize(optimizer, loss):

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


def return_loss_item(optimizer, loss):

    return loss.item()


def standard_epoch(model, dataLoader, loss_function):
    """
    does a standard evaluation epoch
    :param model: model to train
    :param dataLoader: torch dataloader with sources and targets
    :param optimizer: torch optimizer
    :param loss_function: custom loss function
    :return: cumulative loss of all batches
    """
    return standard_frame(model, dataLoader, loss_function, None, function=return_loss_item)


def standard_epoch_optimizer(model, dataLoader, loss_function, optimizer):
    """
    does a standard optimizing epoch
    :param model: model to train
    :param dataLoader: torch dataloader with sources and targets
    :param optimizer: torch optimizer
    :param loss_function: custom loss function
    :return: cumulative loss of all batches
    """

    return standard_frame(model, dataLoader, loss_function, None, function=optimize)
