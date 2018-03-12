import torch.nn.functional as f


def mse_loss(y_input, y_target):
    return f.mse_loss(y_input, y_target)
