import json
from copy import copy, deepcopy
import torch
import numpy as np
import torch.nn as nn
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from base.base_model import BaseModel


class DeepFC(BaseModel):
    def __init__(self):
        super(DeepFC, self).__init__()
        self.fc = None
        self.build_model()

    def build_model(self):
        sz = 8
        self.fc = nn.Sequential(
            nn.Linear(1, sz),
            nn.ReLU(inplace=True),
            nn.Linear(sz, sz),
            nn.ReLU(inplace=True),
            nn.Linear(sz, sz),
            nn.ReLU(inplace=True),
            nn.Linear(sz, sz),
            nn.ReLU(inplace=True),
            nn.Linear(sz, 1)
        )

    def forward(self, x):
        return self.fc(x)


def Jacobian(loss, parameters):
    params = [p for p in parameters]
    J = []
    for p in params:
        g, = torch.autograd.grad(loss, p, create_graph=True)
        sz = list(g.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    J.append(g[i, j].cpu().data.numpy()[0])
        else:
            for i in range(sz[0]):
                J.append(g[i].cpu().data.numpy()[0])
    return np.array(J)


def HessianJacobian(loss, parameters):
    params = [p for p in parameters]
    J, H = [], []
    for p in params:
        g, = torch.autograd.grad(loss, p, create_graph=True)
        sz = list(g.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    J.append(g[i, j])
        else:
            for i in range(sz[0]):
                J.append(g[i])
    for i, g in enumerate(J):
        H.append([])
        for p in params:
            g2, = torch.autograd.grad(g, p, create_graph=True)
            sz = list(g2.size())
            if len(sz) == 2:
                for j in range(sz[0]):
                    for k in range(sz[1]):
                        H[i].append(g2[j, k].cpu().data.numpy()[0])
            else:
                for j in range(sz[0]):
                    H[i].append(g2[j].cpu().data.numpy()[0])
    J = [i.cpu().data.numpy()[0] for i in J]
    return np.array(H), np.array(J)


if __name__ == '__main__':
    loss_ = []
    min_ratio_ = []
    for ti in range(10000):
        if len(loss_) == 100:
            break
        model = DeepFC()
        model.summary()
        optimizer = optim.SGD(model.parameters(), lr=2e-1, momentum=0.9)

        n_data = 128

        def target_func(x):
            return np.sin(4 * np.pi * x) / (4 * np.pi * x)

        x = np.array([i for i in np.linspace(1e-4, 1, n_data)])
        y = np.array([target_func(i) for i in x])
        rand_idx = np.random.permutation(len(x))
        x = np.array([x[i] for i in rand_idx])
        y = np.array([y[i] for i in rand_idx])
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        model.train()
        model.double()
        n_epoch = 1000
        for epoch in range(1, n_epoch):
            data, target = x, y
            data, target = torch.DoubleTensor(data), torch.DoubleTensor(target)
            data, target = Variable(data), Variable(target)

            # H, J = HessianJacobian(loss, model.parameters())
            # eig = np.linalg.eigvals(H)
            # print(np.all(np.isreal(eig)))
            # print(eig)
            output = model(data)
            loss = F.mse_loss(output, target)
            J = Jacobian(loss, model.parameters())
            J = np.expand_dims(J, 0)

            '''
            lr = 6e-1 * (0.5 ** (epoch / 20))
            grad_idx = 0
            for param in model.parameters():
                sz = list(param.size())
                if len(sz) == 2:
                    for i in range(sz[0]):
                        for j in range(sz[1]):
                            param[i, j].data -= lr * J[0][grad_idx]
                            grad_idx += 1
                else:
                    for i in range(sz[0]):
                        param[i].data -= lr * J[0][grad_idx]
                        grad_idx += 1
            '''

            grad_all = np.sum([i ** 2 for i in J.flatten()])
            grad_norm = grad_all ** 0.5
            avg_loss = loss.data[0]
            optimizer.zero_grad()
            if loss.data[0] < 1e-2:
                loss.data[0] = grad_all * 10
            loss.backward()
            optimizer.step()

            if epoch == n_epoch-1:
                sample_loss = []
                for t in range(1000):
                    model_ = deepcopy(model)
                    for param in model_.parameters():
                        sz = list(param.size())
                        if len(sz) == 2:
                            for i in range(sz[0]):
                                for j in range(sz[1]):
                                    param[i, j].data *= 1 + 0.001 * (np.random.rand() - 0.5)
                        else:
                            for i in range(sz[0]):
                                param[i].data *= 1 + 0.001 * (np.random.rand() - 0.5)
                    output = model_(data)
                    loss = F.mse_loss(output, target)
                    sample_loss.append(loss.data[0])

                min_ratio = np.where(np.array(sample_loss) > avg_loss)[0].shape[0] / len(sample_loss)

            print({
                'epoch': epoch,
                'loss': avg_loss,
                'grad_norm': grad_norm
            })

        if grad_norm < 1e-8:
            loss_.append(avg_loss)
            min_ratio_.append(min_ratio)

        print([(_1, _2) for _1, _2 in zip(loss_, min_ratio_)])

    with open('min_ratio.json', 'w') as f:
        json.dump({'loss': loss_, 'min_ratio': min_ratio_}, f)
