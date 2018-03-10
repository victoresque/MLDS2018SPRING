from torch.autograd import Variable
import torch.nn.functional as f


class Trainer:
    def __init__(self, model, data_loader, optimizer, epochs, batch_size):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.data_loader = data_loader

    def train(self):
        for epoch in range(1, self.epochs+1):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
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
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), n_batch * len(data),
                    100.0 * batch_idx / n_batch, loss.data[0]))
