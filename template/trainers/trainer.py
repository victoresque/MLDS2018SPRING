import numpy as np
from torch.autograd import Variable
from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, loss, optimizer, epochs,
                 save_dir, save_freq, resume, with_cuda, logger=None):
        super(Trainer, self).__init__(model, loss, optimizer, epochs,
                                      save_dir, save_freq, resume, logger)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.with_cuda = with_cuda

    def _train_epoch(self, epoch):
        n_batch = len(self.data_loader)
        self.model.train()
        if self.with_cuda:
            self.model.cuda()
        total_loss = 0
        for batch_idx in range(n_batch):
            data, target = next(self.data_loader.next_batch())
            data, target = Variable(data), Variable(target)
            if self.with_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            log_step = int(np.sqrt(self.batch_size))
            if batch_idx % log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_batch * len(data),
                    100.0 * batch_idx / n_batch, loss.data[0]))

        return total_loss / n_batch
