import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
from models.models import DeeperCifarCNN


if __name__ == '__main__':
    base = '../../models/saved/1-3-2/'
    filenames = os.listdir(base)

    plt.figure(figsize=(12, 5))

    for i, name in enumerate(filenames):
        checkpoint = torch.load(base + name)
        logger = checkpoint['logger']
        hidden_size = int(name.split('_')[0][14:])
        model = DeeperCifarCNN(hidden_size)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        plt.subplot(121)
        x = [params]
        y1 = [[entry['loss'] for _, entry in logger.entries.items()][-1]]
        y2 = [[entry['val_loss'] for _, entry in logger.entries.items()][-1]]
        plt.scatter(x, y1, c='#ff8c00', label='train' if not i else None)
        plt.scatter(x, y2, c='b', label='test' if not i else None)
        plt.title('Training/testing loss')
        plt.xlabel('number of parameters')
        plt.ylabel('loss')
        plt.legend(loc='best')

        plt.subplot(122)
        y1 = [[entry['accuracy'] for _, entry in logger.entries.items()][-1]]
        y2 = [[entry['val_accuracy'] for _, entry in logger.entries.items()][-1]]
        plt.scatter(x, y1, c='#ff8c00', label='train' if not i else None)
        plt.scatter(x, y2, c='b', label='test' if not i else None)
        plt.title('Training/testing accuracy')
        plt.xlabel('number of parameters')
        plt.ylabel('accuracy')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
