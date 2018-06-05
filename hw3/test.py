import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model import *
from data_loader import *
from utils.util import ensure_dir, to_variable
import os
import subprocess
import re
import cv2


def main(args):
    checkpoint_path = os.path.join("saved/", args.name, args.checkpoint)
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    embedder = pickle.load(open(os.path.join("saved/", args.name, "embedder.pkl"), 'rb'))
    model = CGAN(config)

    model.load_state_dict(checkpoint['state_dict'])
    with_cuda = not args.no_cuda
    if with_cuda:
        model.cuda()
    model.eval()
    model.summary()

    cond = {'hair': ['black'], 'eyes': ['red']}
    for i in range(25):
        input_noise = torch.randn(1, config['model']['noise_dim'], 1, 1)
        input_cond = embedder.encode_feature(cond)
        input_cond = np.expand_dims(input_cond, 0)
        cond_scale = np.random.uniform(0.8, 1.0, (input_cond.shape[0], 1))
        cond_bias = np.array([np.random.uniform(0, 1 - i[0]) for i in cond_scale]).reshape(cond_scale.shape)
        input_cond = input_cond * cond_scale + cond_bias
        input_noise, input_cond = to_variable(with_cuda, input_noise, input_cond)
        gen_image = model.generator(input_noise, input_cond).cpu().data.numpy()[0]
        gen_image = np.transpose(gen_image, (1, 2, 0))
        cv2.imshow('', (gen_image+1)/2)
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW3-2 Testing')
    parser.add_argument('--name', required=True, type=str,
                        help='Specify the name of folder')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='model checkpoint file name')
    parser.add_argument('--input', required=True, type=str,
                        help='input data')
    parser.add_argument('--output', required=True, type=str,
                        help='output filename')
    parser.add_argument('--no-cuda', action="store_true",
                        help='use CPU instead of GPU')

    args = parser.parse_args()
    main(args)
