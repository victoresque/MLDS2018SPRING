import torch
import numpy as np
import torch.nn as nn
import torch.autograd
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable


class DeepFC(nn.Module):
    def __init__(self):
        super(DeepFC, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    model = DeepFC()
    optim = torch.optim.SGD(model.parameters(), lr=2e-1)

    n_batch = 16
    batch_size = 8

    def target_func(x):
        return np.sin(4 * np.pi * x) / (4 * np.pi * x + 1e-10)

    x = np.array([i for i in np.linspace(0, 1, n_batch * batch_size)])
    y = np.array([target_func(i) for i in x])
    rand_idx = np.random.permutation(len(x))
    x = np.array([x[i] for i in rand_idx])
    y = np.array([y[i] for i in rand_idx])
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    model.train()
    model.cuda()
    for epoch in range(1, 10000):
        total_loss = 0
        for batch_idx in range(n_batch):
            data = x[batch_idx * batch_size:(batch_idx+1) * batch_size]
            target = y[batch_idx * batch_size:(batch_idx+1) * batch_size]
            data, target = torch.FloatTensor(data), torch.FloatTensor(target)
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()

            optim.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward(retain_graph=True)
            optim.step()
            total_loss += loss.data[0]
        avg_loss = total_loss / n_batch

        for p in model.parameters():
            print(p.data, p.grad)

        input()

        print({
            'epoch': epoch,
            'loss': avg_loss
        })
