import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("...")
from models.models import DeepFC, MiddleFC, ShallowFC

if __name__ == '__main__':
    data_list = ['mnist', 'cifar']
    arch_list = ['Deep', 'Middle', 'Shallow']
    base_arch = 'CNN'
    color_list = ['r', 'g', 'b']

    plt.figure(figsize=(12, 9))
    for i, data in enumerate(data_list):
        for arch, color in zip(arch_list, color_list):
            checkpoint = torch.load('../../models/saved/'+arch+data.title()+base_arch+'_'+data+'_checkpoint.pth.tar')
            model = eval(checkpoint['arch'])()
            model.load_state_dict(checkpoint['state_dict'])
            logger = checkpoint['logger']
            '''
            x = []
            y = []
            for _, entry in logger.entries.items():
                x.append(entry['epoch'])
                y.append(entry['loss'])
            x = x[5:]
            y = y[5:]
            plt.subplot(220+i+1)
            plt.title(data + ' loss')
            plt.semilogy(x, y, color, label=arch+base_arch)
            plt.legend(loc="best")
            '''

    plt.tight_layout()
    plt.show()
