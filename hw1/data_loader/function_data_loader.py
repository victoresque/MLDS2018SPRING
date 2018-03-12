import numpy as np
import matplotlib.pyplot as plt
from base.base_data_loader import BaseDataLoader


class FunctionDataLoader(BaseDataLoader):
    def __init__(self, target_func, batch_size, n_sample, x_range):
        super(FunctionDataLoader, self).__init__(batch_size)
        self.batch_size = batch_size
        self.n_sample = n_sample
        self.n_batch = n_sample // batch_size
        self.x_range = x_range
        if target_func == 'sin':
            self.target_func = lambda x: np.sin(4*np.pi*x)
        elif target_func == 'sinc':
            self.target_func = lambda x: np.sin(4*np.pi*x) / (4*np.pi*x)
        elif target_func == 'ceil':
            self.target_func = lambda x: np.ceil(6*x) / 6
        elif target_func == 'damp':
            self.target_func = lambda x: np.exp(-2*x) * np.cos(4*np.pi*x)
        else:
            self.target_func = None

    def next_batch(self):
        for batch_idx in range(self.n_batch):
            x = np.random.uniform(self.x_range[0], self.x_range[1], self.batch_size)
            y = [self.target_func(i) for i in x]
            # plt.plot(x, y, 'ro')
            # plt.show()
            # input()
            x = np.array(x).reshape((-1, 1))
            y = np.array(y).reshape((-1, 1))
            yield (x, y)

    def __len__(self):
        return self.n_batch


