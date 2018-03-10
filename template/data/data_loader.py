import numpy as np
import torch
from torchvision import datasets, transforms


class DataLoader:
    def __init__(self):
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=32, shuffle=True)

    def next_batch(self):
        for batch_idx, (data, target) in enumerate(self.data_loader):
            yield (data, target)

    def __len__(self):
        return len(self.data_loader)
