import torch
from torch.autograd import Variable
from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, loss, optimizer, epochs,
                 save_dir, save_freq, resume):
        super(Trainer, self).__init__(model, loss, optimizer, epochs, save_dir, save_freq, resume)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader

    def _train_epoch(self, epoch):
        n_batch = len(self.data_loader)
        self.model.train()
        for batch_idx in range(n_batch):
            data, target = next(self.data_loader.next_batch())
            data, target = torch.FloatTensor(data), torch.FloatTensor(target)
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.batch_size == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_batch * len(data),
                    100.0 * batch_idx / n_batch, loss.data[0]))
        return 0
