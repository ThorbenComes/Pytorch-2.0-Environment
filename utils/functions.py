import torch

nn = torch.nn

def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.nn.functional.elu(x).add(1)


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return torch.where(x < 1.0, torch.log(x), x.add(-1))


def test_elup1():
    """
    evaluates average error of function and inverse
    :return:
    """
    for i in range(10):
        a = torch.randn((4,4))
        # print(a)
        c = elup1(a)
        # print(c)
        b = elup1_inv(c)
        print(torch.abs(torch.sum(a - b) / 16))

# test_elup1()
