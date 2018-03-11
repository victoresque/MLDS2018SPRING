from torch.autograd import Variable
import torch.nn.functional as f
from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, optimizer, epochs, batch_size,
                 save_dir, save_freq, resume):
        super(Trainer, self).__init__(model, optimizer, epochs, save_dir, save_freq, resume)
        self.batch_size = batch_size
        self.data_loader = data_loader

    def _train_epoch(self, epoch):
        n_batch = len(self.data_loader)

        self.model.train()
        for batch_idx in range(n_batch):
            data, target = next(self.data_loader.next_batch())
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.batch_size == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_batch * len(data),
                    100.0 * batch_idx / n_batch, loss.data[0]))
        return 0
