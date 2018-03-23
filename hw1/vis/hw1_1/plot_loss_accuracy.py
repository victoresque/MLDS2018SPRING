import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    func_list = ['sinc', 'stair']
    arch_list = ['Deep', 'Middle', 'Shallow']
    base_arch = 'FC'
    color_list = ['r', 'g', 'b']
    plt.figure(figsize=(12, 5))
    for i, func in enumerate(func_list):
        for arch, color in zip(arch_list, color_list):
            checkpoint = torch.load(
                '../../models/saved/1-1/' + arch + base_arch + '_' + func + '_checkpoint.pth.tar')
            logger = checkpoint['logger']
            x = [entry['epoch'] for _, entry in logger.entries.items()]
            y = [entry['loss'] for _, entry in logger.entries.items()]
            x = x[150:]
            y = y[150:]
            plt.subplot(120 + i + 1)
            plt.title(func + ' loss')
            plt.semilogy(x, y, color, label=arch + base_arch)
            plt.grid()
            plt.legend(loc="best")

    data_list = ['mnist', 'cifar']
    base_arch = 'CNN'
    plt.figure(figsize=(12, 9))
    for i, data in enumerate(data_list):
        for arch, color in zip(arch_list, color_list):
            checkpoint = torch.load('../../models/saved/1-1/'+arch+data.title()+base_arch+'_'+data+'_checkpoint.pth.tar')
            logger = checkpoint['logger']
            x = [entry['epoch'] for _, entry in logger.entries.items()]
            y1 = [entry['loss'] for _, entry in logger.entries.items()]
            y2 = [entry['accuracy'] for _, entry in logger.entries.items()]
            x = x[5:]
            y1 = y1[5:]
            y2 = y2[5:]
            plt.subplot(220 + i + 1)
            plt.title(data + ' loss')
            plt.semilogy(x, y1, color, label=arch + data.title() + base_arch)
            plt.grid()
            plt.legend(loc="best")
            plt.subplot(220 + i + 3)
            plt.title(data + ' accuracy')
            plt.plot(x, y2, color, label=arch + data.title() + base_arch)
            plt.grid()
            plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
