import os
import torch
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from models.models import DeepMnistCNN, ShallowMnistCNN
from sklearn.externals import joblib


if __name__ == '__main__':
    base = 'models/saved/1-2-1/'
    epoch_step = 4
    epoch_max = 40
    cmap = get_cmap('hsv')
    plt.figure(figsize=(16, 7))

    # All layers
    all_params = []
    all_accs = []
    for i in range(1, 9):
        checkpoint_base = os.path.join(base, str(i))
        checkpoint_filenames = sorted(os.listdir(checkpoint_base))
        for epoch in range(2, epoch_max, epoch_step):
            filename = checkpoint_filenames[epoch]
            checkpoint = torch.load(os.path.join(checkpoint_base, filename))
            model = eval(checkpoint['arch'])()
            model.load_state_dict(checkpoint['state_dict'])
            params = np.zeros((0, ))
            for p in model.parameters():
                params = np.append(params, p.cpu().data.numpy().flatten())
            all_params.append(params)
            all_accs.append(checkpoint['logger'].entries[epoch-1]['accuracy'])

    all_params = np.array(all_params)
    IncrementalPCA(2, batch_size=6).fit_transform(all_params)
    x = [p[0] for p in all_params]
    y = [p[1] for p in all_params]

    seg = len(range(1, epoch_max, epoch_step))
    plt.subplot(121)
    plt.title('All layers (DeepMnistCNN)')
    for i in range(1, 9):
        x_ = x[(i - 1) * seg:i * seg]
        y_ = y[(i - 1) * seg:i * seg]
        a_ = all_accs[(i - 1) * seg:i * seg]
        plt.plot(x_, y_, 'o:', color=cmap((i - 1) / 7))
        for xi, yi, ai in zip(x_, y_, a_):
            plt.annotate(str('{:.1f}'.format(ai * 100)), xy=(xi, yi),
                         xytext=(xi + 0.004, yi + 0.004), color=cmap((i - 1) / 7.5))
    plt.grid()

    # One layer
    all_params = []
    all_accs = []
    for i in range(1, 9):
        checkpoint_base = os.path.join(base, str(i))
        checkpoint_filenames = sorted(os.listdir(checkpoint_base))
        for epoch in range(2, epoch_max, epoch_step):
            filename = checkpoint_filenames[epoch]
            checkpoint = torch.load(os.path.join(checkpoint_base, filename))
            model = eval(checkpoint['arch'])()
            model.load_state_dict(checkpoint['state_dict'])
            params = model.state_dict()['cnn.3.weight'].numpy().flatten()
            params = np.append(params, model.state_dict()['cnn.3.bias'].numpy().flatten())
            all_params.append(params)
            all_accs.append(checkpoint['logger'].entries[epoch - 1]['accuracy'])

    all_params = np.array(all_params)
    IncrementalPCA(2, batch_size=12).fit_transform(all_params)
    x = [p[0] for p in all_params]
    y = [p[1] for p in all_params]

    seg = len(range(1, epoch_max, epoch_step))
    plt.subplot(122)
    plt.title('Second CONV layer (DeepMnistCNN)')
    for i in range(1, 9):
        x_ = x[(i - 1) * seg:i * seg]
        y_ = y[(i - 1) * seg:i * seg]
        a_ = all_accs[(i - 1) * seg:i * seg]
        plt.plot(x_, y_, 'o:', color=cmap((i - 1) / 7))
        for xi, yi, ai in zip(x_, y_, a_):
            plt.annotate(str('{:.1f}'.format(ai * 100)), xy=(xi, yi),
                         xytext=(xi + 0.004, yi + 0.004), color=cmap((i - 1) / 7.5))
    plt.grid()
    plt.show()

