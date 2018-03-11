import torch
from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size):
        super(DataLoader, self).__init__(batch_size)
        self.data_dir = data_dir
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=self.batch_size, shuffle=True)

    def next_batch(self):
        for batch_idx, (data, target) in enumerate(self.data_loader):
            yield (data, target)

    def __len__(self):
        return len(self.data_loader)
