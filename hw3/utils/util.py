import os
import numpy as np
import torch
import torchvision
import cv2
from torch.autograd import Variable


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_metrics(metrics, output, target):
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    output = np.array([1 if i > 0.5 else 0 for i in output])
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics


def to_variable(with_cuda, *args):
    return_var = []
    for data in args:
        if isinstance(data, np.ndarray):
            return_var.append(Variable(torch.FloatTensor(data)))
        else:
            return_var.append(Variable(data))
    if with_cuda:
        return_var = [data.cuda() for data in return_var]
    return return_var if len(return_var) > 1 else return_var[0]


def show_grid(images):
    images = (images.cpu().data[:64] + 1) / 2
    grid = torchvision.utils.make_grid(images).numpy()
    grid = np.transpose(grid, (1, 2, 0))
    cv2.imshow('generated images', grid)
    cv2.waitKey(1)
