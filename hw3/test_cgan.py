import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model import *
from data_loader import *
from utils.util import ensure_dir, to_variable, save_imgs
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

    conds = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.split(',')[1]
            line = line.split()
            conds.append({'hair': [line[0]], 'eyes': [line[2]]})

    model.load_state_dict(checkpoint['state_dict'])
    with_cuda = not args.no_cuda
    if with_cuda:
        model.cuda()
    model.eval()
    model.summary()

    gen_images = []
    torch.manual_seed(7777777)
    for cond in conds:
        input_noise = torch.randn(1, config['model']['noise_dim'], 1, 1)
        input_cond = embedder.encode_feature(cond)
        input_cond = np.expand_dims(input_cond, 0) * 0.9 + 0.05
        input_noise, input_cond = to_variable(with_cuda, input_noise, input_cond)
        gen_image = model.generator(input_noise, input_cond).cpu().data.numpy()[0]
        gen_image = np.transpose(gen_image, (1, 2, 0))
        gen_image = (gen_image+1)/2
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
        gen_images.append(gen_image)

    save_imgs(np.array(gen_images), args.output)


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
