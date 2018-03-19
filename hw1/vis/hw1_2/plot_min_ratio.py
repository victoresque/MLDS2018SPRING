from copy import copy
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
    J = []
    H = []
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
    model = DeepFC()
    optim = torch.optim.Adam(model.parameters())

    n_batch = 1
    batch_size = 64

    def target_func(x):
        return np.sin(4 * np.pi * x)

    x = np.array([i for i in np.linspace(0, 1, n_batch * batch_size)])
    y = np.array([target_func(i) for i in x])
    rand_idx = np.random.permutation(len(x))
    x = np.array([x[i] for i in rand_idx])
    y = np.array([y[i] for i in rand_idx])
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    model.train()
    model.double()
    for epoch in range(1, 10000):
        total_loss = 0
        for batch_idx in range(n_batch):
            data = x[batch_idx * batch_size:(batch_idx+1) * batch_size]
            target = y[batch_idx * batch_size:(batch_idx+1) * batch_size]
            data, target = torch.DoubleTensor(data), torch.DoubleTensor(target)
            data, target = Variable(data), Variable(target)
            # data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = F.mse_loss(output, target)

            H, J = HessianJacobian(loss, model.parameters())
            J = np.expand_dims(J, 0)

            lr = 1e-1 * (0.5 ** (epoch / 20))
            grad_idx = 0
            for param in model.parameters():
                sz = list(param.size())
                if len(sz) == 2:
                    for i in range(sz[0]):
                        for j in range(sz[1]):
                            param[i, j].data -= lr * J[0][grad_idx]
                            # param[i, j].data -= delta[grad_idx][0]
                            grad_idx += 1
                else:
                    for i in range(sz[0]):
                        param[i].data -= lr * J[0][grad_idx]
                        # param[i].data -= delta[grad_idx][0]
                        grad_idx += 1

            total_loss += loss.data[0]

        grad_all = np.sum([i**2 for i in J.flatten()])
        grad_norm = grad_all ** 0.5
        print(grad_norm)
        avg_loss = total_loss / n_batch
        print({
            'epoch': epoch,
            'loss': avg_loss
        })
