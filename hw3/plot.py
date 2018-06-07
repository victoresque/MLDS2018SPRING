import sys
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    log = torch.load(sys.argv[1])['logger'].entries

    n_epoch = len(log)
    iter_per_epoch = len(log[1]['full_loss'])
    epoch_loss_d = []
    epoch_loss_g = []
    iter_loss_d = []
    iter_loss_g = []

    for i in range(1, n_epoch+1):
        epoch_loss_d.append(log[i]['loss_d'])
        epoch_loss_g.append(log[i]['loss_g'])
        for j in range(iter_per_epoch):
            iter_loss_d.append(log[i]['full_loss'][j]['loss_d'])
            if log[i]['full_loss'][j]['loss_g'] is not None:
                iter_loss_g.append(log[i]['full_loss'][j]['loss_g'])

    plt.figure(figsize=(8, 6))

    iter_loss_d_avg = []
    iter_loss_g_avg = []

    K = float(sys.argv[2]) if len(sys.argv) == 3 else 5
    for i in range(len(iter_loss_d)//K):
        s = 0
        for j in range(K):
            s = s + iter_loss_d[i*K + j]
        iter_loss_d_avg.append(s/K)

    for i in range(len(iter_loss_g)//K):
        s = 0
        for j in range(K):
            s = s + iter_loss_g[i*K + j]
        iter_loss_g_avg.append(s/K)

    total_iter = n_epoch*iter_per_epoch
    plt.plot([i/(len(iter_loss_d)//K)*total_iter for i in range(1, len(iter_loss_d)//K+1)], iter_loss_d_avg, '#FFAAAA')
    plt.plot([i/(len(iter_loss_g)//K)*total_iter for i in range(1, len(iter_loss_g)//K+1)], iter_loss_g_avg, '#AAAAFF')
    plt.plot([i*iter_per_epoch for i in range(1, n_epoch+1)], epoch_loss_d, 'r')
    plt.plot([i*iter_per_epoch for i in range(1, n_epoch+1)], epoch_loss_g, 'b')
    plt.xlim((0, total_iter))
    plt.show()
