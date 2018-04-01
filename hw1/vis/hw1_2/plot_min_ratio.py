from copy import copy
import torch
import numpy as np
import torch.nn as nn
import torch.autograd
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable
from numpy.linalg import eig


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
    
def GradNorm(model):
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    grad_norm = grad_all ** 0.5
    return grad_norm


def Jacobian(loss, parameters):
    params = [p for p in parameters]
    J = []
    for p in params:
        g, = torch.autograd.grad(loss, p, create_graph=True) # d(loss) / d(weight at according layer) shape == p.shape
        sz = list(g.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    J.append(g[i, j].cpu().data.numpy()[0])
        else:
            assert(len(sz) == 1)
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

def train(run):
    model = DeepFC()
    optim = torch.optim.Adam(model.parameters())
    optim_grad = torch.optim.Adam(model.parameters())
    early_stop_count = 0

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
    for epoch in range(1, 20001):
        total_loss = 0
        for batch_idx in range(n_batch):
            data = x[batch_idx * batch_size:(batch_idx+1) * batch_size]
            target = y[batch_idx * batch_size:(batch_idx+1) * batch_size]
            data, target = torch.FloatTensor(data), torch.FloatTensor(target)
            data, target = Variable(data), Variable(target)
            # data, target = data.cuda(), target.cuda()

            optim.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            
        avg_loss = total_loss / n_batch
        grad_norm = GradNorm(model)
        print({
            'epoch': epoch,
            'loss': avg_loss,
            'grad_norm': grad_norm
        })
        
    
    for epoch in range(20001, 22001):
        total_loss = 0
        total_loss_grad = 0
        for batch_idx in range(n_batch):
            
            
            data = x[batch_idx * batch_size:(batch_idx+1) * batch_size]
            target = y[batch_idx * batch_size:(batch_idx+1) * batch_size]
            data, target = torch.FloatTensor(data), torch.FloatTensor(target)
            data, target = Variable(data), Variable(target)
            # data, target = data.cuda(), target.cuda()
    
            output = model(data)
            loss = F.mse_loss(output, target)
            total_loss += loss.data[0]
            
            optim_grad.zero_grad()
            grad_list = []
            for p in model.parameters():
                g, = torch.autograd.grad(loss, p, create_graph=True) # d(loss) / d(weight at according layer) shape == p.shape
                sz = list(g.size())
                if len(sz) == 2:
                    for i in range(sz[0]):
                        for j in range(sz[1]):
                            grad_list.append(g[i, j])
                else:
                    assert(len(sz) == 1)
                    for i in range(sz[0]):
                        grad_list.append(g[i])
                
            grads = torch.cat(grad_list)
            zeros = torch.zeros_like(grads)
            loss_grad = F.mse_loss(grads, zeros)
            #loss_grad = torch.sum(grads**2)
            
            loss_grad.backward()
            optim_grad.step()
            
            total_loss_grad += loss_grad.data[0]
            output = model(data)
            loss = F.mse_loss(output, target)
            H, J = HessianJacobian(loss, model.parameters())
            try:
                w, v = eig(H)
            except:
                w ,v = None, None
                
            if w is not None:
                positive_count = np.sum([1 if i > 0 else 0 for i in w])
                min_ratio = positive_count / len(w)
            else:
                min_ratio = -1
            

        grad_norm = GradNorm(model)
        avg_loss = total_loss / n_batch
        avg_loss_grad = total_loss_grad / n_batch
        log = {
            'epoch': epoch,
            'loss': avg_loss,
            'avg_loss_grad': avg_loss_grad,
            'grad_norm': grad_norm,
            'min_ratio': min_ratio
        }
        print('RUN {}'.format(run), end='')
        print(log)
        
        if early_stop_count >= 10 or epoch == 22000:
            return log
        
        if grad_norm < 1e-3:
            early_stop_count += 1
        
        


if __name__ == '__main__':
    
    min_ratio, loss, grad_norm = [], [], []
    for i in range(100):
        print("RUN {}".format(i))
        log = train(i)
        min_ratio.append(log['min_ratio'])
        loss.append(log['loss'])
        grad_norm.append(log['grad_norm'])
    
    
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.scatter(min_ratio, loss)
    plt.title('min_ratio vs loss')
    plt.xlabel('min_ratio')
    plt.ylabel('loss')
    plt.grid()
    
    plt.subplot(122)
    plt.scatter(grad_norm, loss)
    plt.title('grad_norm vs loss')
    plt.xlabel('grad_norm')
    plt.ylabel('loss')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('./min_ratio.png')
    
        
        
