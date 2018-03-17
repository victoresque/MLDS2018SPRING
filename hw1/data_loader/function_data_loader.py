import random
import numpy as np
import matplotlib.pyplot as plt
from base.base_data_loader import BaseDataLoader


class FunctionDataLoader(BaseDataLoader):
    def __init__(self, target_func, batch_size, n_sample, x_range, shuffle=True):
        super(FunctionDataLoader, self).__init__(batch_size)
        self.batch_size = batch_size
        self.n_batch = n_sample // batch_size
        self.x_range = x_range
        if target_func == 'sin':
            self.target_func = lambda x: np.sin(4*np.pi*x)
        elif target_func == 'sinc':
            self.target_func = lambda x: np.sin(4*np.pi*x) / (4*np.pi*x + 1e-10)
        elif target_func == 'stair':
            self.target_func = lambda x: np.ceil(4*x) / 4 - 2.5
        elif target_func == 'damp':
            self.target_func = lambda x: np.exp(-2*x) * np.cos(4*np.pi*x)
        else:
            self.target_func = None
        self.__generate_data()
        if shuffle:
            rand_idx = np.random.permutation(len(self.x))
            self.x = np.array([self.x[i] for i in rand_idx])
            self.y = np.array([self.y[i] for i in rand_idx])
        self.batch_idx = 0

    def __generate_data(self):
        self.x = np.array([i for i in np.linspace(self.x_range[0], self.x_range[1],
                                                  self.n_batch * self.batch_size)])
        self.y = np.array([self.target_func(i) for i in self.x])

    def next_batch(self):
        x_batch = self.x[self.batch_idx * self.batch_size:(self.batch_idx+1) * self.batch_size]
        y_batch = self.y[self.batch_idx * self.batch_size:(self.batch_idx+1) * self.batch_size]
        x_batch = x_batch.reshape((-1, 1))
        y_batch = y_batch.reshape((-1, 1))
        self.batch_idx = self.batch_idx+1 if self.batch_idx != self.n_batch-1 else 0
        return x_batch, y_batch

    def __len__(self):
        return self.n_batch
