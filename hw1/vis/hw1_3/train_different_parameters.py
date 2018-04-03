import os
import subprocess

os.chdir('../../')

if __name__ == '__main__':
    for i in range(8, 65, 2):
        subprocess.call(['python', 'main.py',
                         '--batch-size', '128',
                         '--epochs', '200',
                         '--save-freq', '200',
                         '--dataset', 'cifar',
                         '--arch', 'deeper'+str(i * 2),
                         '--validation-split', '0.1',
                         '--save-dir', 'models/saved/1-3-2'])

