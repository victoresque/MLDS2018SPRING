import argparse
import torch
import numpy as np
import sys, os
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
sys.path.append('../../')
from base.base_trainer import BaseTrainer
from models.models import DeepMnistCNN 
from models.models import DeepCifarCNN 
from data_loader.data_loader import MnistLoader, CifarLoader
from utils.util import split_validation
from logger.logger import Logger

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
        

def main():
    parser = argparse.ArgumentParser(description='hw1-3-3 sharpness plot')
    parser.add_argument('-b', '--batch', default=50, type=int,
                        help='batch size to computing loss (default: 128)')
    parser.add_argument('--dataset', default='cifar', type=str,
                        help='choose dataset to plot sharpness [mnist, cifar] (default: mnist)')
    parser.add_argument('--cuda', action='store_true', help='use gpu')
    args = parser.parse_args()


    model_name = 'Deep' + args.dataset.title() + 'CNN'
    base = '../../models/saved/1-3-3/bonus/' + args.dataset
    plot_files_path = []
    checkpoints_batch_size = []
    for folder_name in sorted(os.listdir(base), key=lambda s: int(s[s.find('batch')+5:])):
        checkpoints_batch_size.append(int(folder_name[folder_name.find('batch')+5:]))
        folder_path = os.path.join(base, folder_name)
        files = sorted(os.listdir(folder_path), key=lambda s: int(s[s.find('epoch')+5: s.rfind('.pth')]))
        plot_files_path.append(os.path.join(folder_path, files[-1]))

    checkpoints_state_dict = []
    checkpoints_acc, checkpoints_val_acc = [], []
    checkpoints_loss, checkpoints_val_loss = [], []
    for file_path in plot_files_path:
        checkpoint = torch.load(file_path, map_location='cpu')
        epoch = checkpoint['epoch']
        checkpoints_state_dict.append(checkpoint['state_dict'])
        checkpoints_loss.append(checkpoint['logger'].entries[epoch]['loss'])
        checkpoints_val_loss.append(checkpoint['logger'].entries[epoch]['val_loss'])
        checkpoints_acc.append(checkpoint['logger'].entries[epoch]['accuracy'])
        checkpoints_val_acc.append(checkpoint['logger'].entries[epoch]['val_accuracy'])

    data_loader = eval(args.dataset.title() + 'Loader')(args.batch)
    print('Starting Computing sharpness ...', end='')
    checkpoints_sharpness = []
    print('Model count:' , len(checkpoints_state_dict))
    for state_dict in checkpoints_state_dict:
        checkpoints_sharpness.append(compute_sharpness(model_name, state_dict, data_loader, args))
    print('\nsharpness Computing Completed ')
    
    checkpoints_batch_size = np.array(checkpoints_batch_size)
    checkpoints_acc = np.array(checkpoints_acc)
    checkpoints_val_acc = np.array(checkpoints_val_acc)
    checkpoints_loss = np.array(checkpoints_loss)
    checkpoints_val_loss = np.array(checkpoints_val_loss)
    checkpoints_sharpness = np.array(checkpoints_sharpness)


    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1,2,1)
    ln1 = ax1.semilogx(checkpoints_batch_size, checkpoints_loss, 'b-', label='training loss')
    ln2 = ax1.semilogx(checkpoints_batch_size, checkpoints_val_loss, 'b--', label='validation loss')
    ax2 = ax1.twinx()
    ln3 = ax2.loglog(checkpoints_batch_size, checkpoints_sharpness, 'r-', label='sharpness')
    ax1.set_xlabel('batch size')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('cross entropy', color='b')
    ax1.tick_params('y', colors='b')
    ax2.set_ylabel('sharpness', color='r')
    ax2.tick_params('y', colors='r')

    lns1 = ln1+ln2+ln3
    labs1 = [l.get_label() for l in lns1]
    ax1.set_title(args.dataset.upper() + ' loss vs. sharpness')

    ax3 = fig.add_subplot(1,2,2)
    ln4 = ax3.semilogx(checkpoints_batch_size, checkpoints_acc, 'b-', label='training accuracy')
    ln5 = ax3.semilogx(checkpoints_batch_size, checkpoints_val_acc, 'b--', label='validation accuracy')
    ax4 = ax3.twinx()
    ln6 = ax4.loglog(checkpoints_batch_size, checkpoints_sharpness, 'r-', label='sharpness')
    ax1.set_xlabel('batch size')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax3.set_ylabel('accuracy', color='b')
    ax3.tick_params('y', colors='b')
    ax4.set_ylabel('sharpness', color='r')
    ax4.tick_params('y', colors='r')

    lns2 = ln4+ln5+ln6
    labs2 = [l.get_label() for l in lns2]
    ax3.set_title(args.dataset.upper() + ' accuracy vs. sharpness')
    ax1.legend(lns1, labs1, loc=0)
    ax3.legend(lns2, labs2, loc=0)
    ax1.set_ylim((0, 1.5))
    ax3.set_ylim((0.4, 0.9))
    fig.tight_layout()
    savefig_path = './{}_sharpness.png'.format(args.dataset)
    fig.savefig(savefig_path)
    print("saving file : {}".format(savefig_path))


def compute_sharpness(model_name, state_dict, data_loader, args):
    test_model = eval(model_name)()
    test_model.load_state_dict(state_dict)
    max_sharpness = 0
    epsilon = 1e-2

    for batch_idx, (data, target) in enumerate(data_loader):
        target_dtype = str(target.dtype)
        data = torch.FloatTensor(data)
        if args.cuda:
            test_model.cuda()
            data = Variable(data.cuda(), requires_grad=True)
        else:
            data = Variable(data, requires_grad=True)
            
        output = test_model(data)
        output_dim = output.size()[1]

        loss0 = F.cross_entropy(output, Variable(torch.LongTensor(target)))
        
        model = test_model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        rand_ball = np.random.uniform(size=(n_params, ))
        rand_ball = rand_ball / np.linalg.norm(rand_ball) * epsilon

        p_idx = 0
        params = [p for p in model.parameters()]
        for p in params:
            sz = list(p.size())
            if len(sz) == 2:
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        p[i, j].data += rand_ball[p_idx]
                        p_idx += 1
            else:
                for i in range(sz[0]):
                    p[i].data += rand_ball[p_idx]
                    p_idx += 1

        output = test_model(data)
        loss1 = F.cross_entropy(output, Variable(torch.LongTensor(target)))
        sharpness = (loss1.data[0] - loss0.data[0]) / (1 + loss0.data[0])
        max_sharpness = max(sharpness, max_sharpness)

    print('------------------')
    return max_sharpness
 

if __name__ == '__main__':
    main()


