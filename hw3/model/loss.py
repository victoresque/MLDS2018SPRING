import torch.nn.functional as F
from torch import nn

def binary_crossentropy(y_input, y_target):
    loss = nn.BCELoss()
    return loss(y_input, y_target)
