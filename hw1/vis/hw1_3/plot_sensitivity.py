import argparse
import torch
import numpy as np
import sys, os
import torch.optim as optim
from torch.autograd import Variable, grad

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
    parser = argparse.ArgumentParser(description='hw1-3-3 sensitivity plot')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='batch size to computing loss (default: 128)')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='choose dataset to plot sensitivity [mnist, cifar] (default: mnist)')
    parser.add_argument('--cuda', action='store_true', help='use gpu')
    args = parser.parse_args()


    model_name = 'Deep' + args.dataset.title() + 'CNN'
    base = 'models/saved/1-3-3/part2/' + args.dataset
    plot_files_path = []
    checkpoints_batch_size = []
    for folder_name in sorted(os.listdir(base), key=lambda s: int(s[s.find('batch')+5:])):
        checkpoints_batch_size.append(int(folder_name[folder_name.find('batch')+5:]))
        folder_path = os.path.join(base, folder_name)
        files = sorted(os.listdir(folder_path), key=lambda s: int(s[s.find('epoch')+5: s.rfind('_loss')]))
        plot_files_path.append(os.path.join(folder_path, files[-1]))

    checkpoints_state_dict = []
    checkpoints_acc, checkpoints_val_acc = [], []
    checkpoints_loss, checkpoints_val_loss = [], []
    for file_path in plot_files_path:
        checkpoint = torch.load(file_path)
        epoch = checkpoint['epoch']
        checkpoints_state_dict.append(checkpoint['state_dict'])
        checkpoints_loss.append(checkpoint['logger'].entries[epoch]['loss'])
        checkpoints_val_loss.append(checkpoint['logger'].entries[epoch]['val_loss'])
        checkpoints_acc.append(checkpoint['logger'].entries[epoch]['accuracy'])
        checkpoints_val_acc.append(checkpoint['logger'].entries[epoch]['val_accuracy'])

    data_loader = eval(args.dataset.title() + 'Loader')(args.batch)
    print('Starting Computing Sensitivity ...', end='')
    checkpoints_sensitivity = []
    for state_dict in checkpoints_state_dict:
        checkpoints_sensitivity.append(compute_sensitivity(model_name, state_dict, data_loader, args))
    print('\nSensitivity Computing Completed ')
    
    checkpoints_batch_size = np.array(checkpoints_batch_size)
    checkpoints_acc = np.array(checkpoints_acc)
    checkpoints_val_acc = np.array(checkpoints_val_acc)
    checkpoints_loss = np.array(checkpoints_loss)
    checkpoints_val_loss = np.array(checkpoints_val_loss)
    checkpoints_sensitivity = np.array(checkpoints_sensitivity)


    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1,2,1)
    ln1 = ax1.semilogx(checkpoints_batch_size, checkpoints_loss, 'b-', label='training loss')
    ln2 = ax1.semilogx(checkpoints_batch_size, checkpoints_val_loss, 'b--', label='validation loss')
    ax2 = ax1.twinx()
    ln3 = ax2.loglog(checkpoints_batch_size, checkpoints_sensitivity, 'r-', label='sensitivity')
    ax1.set_xlabel('batch size')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('cross entropy', color='b')
    ax1.tick_params('y', colors='b')
    ax2.set_ylabel('sensitivity', color='r')
    ax2.tick_params('y', colors='r')

    lns1 = ln1+ln2+ln3
    labs1 = [l.get_label() for l in lns1]
    ax1.legend(lns1, labs1, loc=0)
    ax1.set_title(args.dataset.upper() + ' loss vs. sensitivity')

    ax3 = fig.add_subplot(1,2,2)
    ln4 = ax3.semilogx(checkpoints_batch_size, checkpoints_acc, 'b-', label='training accuracy')
    ln5 = ax3.semilogx(checkpoints_batch_size, checkpoints_val_acc, 'b--', label='validation accuracy')
    ax4 = ax3.twinx()
    ln6 = ax4.loglog(checkpoints_batch_size, checkpoints_sensitivity, 'r-', label='sensitivity')
    ax1.set_xlabel('batch size')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax3.set_ylabel('accuracy', color='b')
    ax3.tick_params('y', colors='b')
    ax4.set_ylabel('sensitivity', color='r')
    ax4.tick_params('y', colors='r')

    lns2 = ln4+ln5+ln6
    labs2 = [l.get_label() for l in lns2]
    ax3.legend(lns2, labs2, loc=0)
    ax3.set_title(args.dataset.upper() + ' accuracy vs. sensitivity')
    
    fig.tight_layout()
    savefig_path = './{}_sensitivity.png'.format(args.dataset)
    fig.savefig(savefig_path)
    print("saving file : {}".format(savefig_path))


def compute_sensitivity(model_name, state_dict, data_loader, args):
    test_model = eval(model_name)()
    test_model.load_state_dict(state_dict)
    total_sensitivity = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx % 16 == 0:
            sys.stdout.write('\b\b\b   ')
        if batch_idx % 16 == 4:
            sys.stdout.write('\b\b\b.  ')
        if batch_idx % 16 == 8:
            sys.stdout.write('\b\b\b.. ')
        if batch_idx % 16 == 12:
            sys.stdout.write('\b\b\b...')
        sys.stdout.flush()    
        target_dtype = str(target.dtype)
        data = torch.FloatTensor(data)
        #target = torch.FloatTensor(target) if target_dtype[0] == 'f' else torch.LongTensor(target)
        if args.cuda:
            #data, target = Variable(data.cuda(), requires_grad=True), Variable(target.cuda())
            test_model.cuda()
            data = Variable(data.cuda(), requires_grad=True)
        else:
            #data, target = Variable(data, requires_grad=True), Variable(target)
            data = Variable(data, requires_grad=True)
            
        output = test_model(data)
        output_dim = output.size()[1]

        Jacobian_matrix = []
        for i in range(output_dim):
            independent_sum = torch.sum(output[:,i])
            g, = grad(independent_sum, data, retain_graph=True)
            gradient_vector = g.data.cpu().numpy()
            Jacobian_matrix.append(gradient_vector.flatten())
        
        Jacobian_matrix = np.array(Jacobian_matrix)
        total_sensitivity += np.sum(Jacobian_matrix ** 2)

    return total_sensitivity ** 0.5
 

if __name__ == '__main__':
    main()


