import os
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    checkpoint_base = '/home/victorhuang/Desktop/MLDS2018SPRING/hw1/models/saved/1-2-2/'
    checkpoint_filenames = sorted(os.listdir(checkpoint_base))
    color_list = ['r', 'b']
    plt.figure(figsize=(12, 9))
    for i, (filename, color) in enumerate(zip(checkpoint_filenames, color_list)):
        checkpoint = torch.load(os.path.join(checkpoint_base, filename))
        logger = checkpoint['logger']
        x = [entry['epoch'] for _, entry in logger.entries.items()]
        y = [entry['loss'] for _, entry in logger.entries.items()]
        x = x[2:int(checkpoint['epoch'] * 0.6)]
        y = y[2:int(checkpoint['epoch'] * 0.6)]
        plt.subplot(220 + i + 1)
        plt.title('' + ' loss')
        plt.plot(x, y, color, label='loss')
        plt.legend(loc="best")

        x = [entry['epoch'] for _, entry in logger.entries.items()]
        y = [entry['grad_norm'] for _, entry in logger.entries.items()]
        x = x[2:int(checkpoint['epoch'] * 0.6)]
        y = y[2:int(checkpoint['epoch'] * 0.6)]
        plt.subplot(220 + i + 3)
        plt.title('' + ' gradient norm')
        plt.plot(x, y, color, label='grad_norm')
        plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
