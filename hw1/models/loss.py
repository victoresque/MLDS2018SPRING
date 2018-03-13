import torch.nn.functional as f


def mse_loss(pred, target):
    return f.mse_loss(pred, target)


def cross_entropy_loss(pred, target):
    return f.cross_entropy(pred, target)
