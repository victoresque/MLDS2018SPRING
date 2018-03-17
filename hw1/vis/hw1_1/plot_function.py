import sys
sys.path.append("...")
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from models.models import DeepFC, MiddleFC, ShallowFC

if __name__ == '__main__':
    func_name_list = ['sinc', 'stair']
    func_lambda_list = [lambda x: np.sin(4*np.pi*x) / (4*np.pi*x + 1e-10),
                        lambda x: np.ceil(4 * x) / 4 - 2.5]
    arch_list = ['Deep', 'Middle', 'Shallow']
    base_arch = 'FC'
    color_list = ['r', 'g', 'b']

    plt.figure(figsize=(12, 9))
    for i, func_name in enumerate(func_name_list):
        func = func_lambda_list[i]
        x = np.array([i for i in np.linspace(0, 1, 10000)])
        y_target = np.array([func(i) for i in x])
        plt.title(func_name+' loss')
        plt.subplot(220 + i + 1)
        plt.plot(x, y_target, 'k', label='Ground truth')
        for arch, color in zip(arch_list, color_list):
            checkpoint = torch.load('../../models/saved/'+arch+base_arch+'_'+func_name+'_checkpoint.pth.tar')
            model = eval(checkpoint['arch'])()
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            y_pred = np.array([model(Variable(torch.FloatTensor(np.array([[i]])))).data.numpy() for i in x]).squeeze()
            plt.plot(x, y_pred, color, label=arch+base_arch)
            plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
