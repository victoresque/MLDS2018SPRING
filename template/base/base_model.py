import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None

    def build_model(self):
        raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)
