from copy import copy
import torch
import numpy as np
from base.base_data_loader import BaseDataLoader

# TODO: Dataset loading
# TODO: Word embedding/One-hot encoding
# TODO: Tokens (<PAD>, <BOS>, <EOS>, <UNK>, ...)


class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True):
        super(DataLoader, self).__init__(batch_size)
        self.data_dir = data_dir
        self.x = []
        self.y = []
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.n_batch = len(self.x) // self.batch_size
        self.batch_idx = 0
        self.shuffle = shuffle

    def __iter__(self):
        self.n_batch = len(self.x) // self.batch_size
        self.batch_idx = 0
        assert self.n_batch > 0
        if self.shuffle:
            rand_idx = np.random.permutation(len(self.x))
            self.x = np.array([self.x[i] for i in rand_idx])
            self.y = np.array([self.y[i] for i in rand_idx])
        return self

    def __next__(self):
        if self.batch_idx < self.n_batch:
            x_batch = self.x[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            y_batch = self.y[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            self.batch_idx = self.batch_idx + 1
            return x_batch, y_batch
        else:
            raise StopIteration

    def __len__(self):
        """
        :return: Total batch number
        """
        self.n_batch = len(self.x) // self.batch_size
        return self.n_batch

