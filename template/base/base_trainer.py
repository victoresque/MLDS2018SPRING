import torch
import os
from tqdm import tqdm
import numpy as np


class BaseTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        for cur_epoch in range(self.config.cur_epoch, self.config.num_epochs, 1):
            self.__train_epoch()

    def __train_epoch(self):
        pass



