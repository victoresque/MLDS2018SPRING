import os
import math
import shutil
import torch
from tqdm import tqdm
from utils.util import ensure_dir


class BaseTrainer:
    def __init__(self, model, loss, optimizer, epochs,
                 save_dir, save_freq, resume, logger=None):
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.optimizer = optimizer
        self.min_loss = math.inf
        self.start_epoch = 1
        self.logger = logger
        ensure_dir(save_dir)
        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            loss = self._train_epoch(epoch)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, loss)
            if self.logger:
                self.logger.add_entry(epoch, {
                    'loss': loss
                })
                self.logger.print()

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, loss):
        if loss < self.min_loss:
            self.min_loss = loss
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'min_loss': self.min_loss,
            'logger': self.logger
        }
        filename = os.path.join(self.save_dir, 'checkpoint_epoch{:02d}.pth.tar'.format(epoch))
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        if loss == self.min_loss:
            shutil.copyfile(filename, os.path.join(self.save_dir, 'model_best.pth.tar'))

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.min_loss = checkpoint['min_loss']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
