import torch.nn.functional as f


def cross_entropy(y_input, y_target):
    return f.cross_entropy(y_input, y_target)
