import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.seq2seq import Seq2Seq
from data_loader import *
from preprocess.embedding import *
from utils.util import ensure_dir
from utils.beam_search import beam_search
import os, subprocess
import re


def main(args):
    checkpoint_path = os.path.join("saved/", args.name, args.checkpoint)
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    embedder = eval(config['embedder']['type'])
    embedder_path = os.path.join("saved/", args.name, "embedder.pkl")
    data_loader = ChatbotDataLoader(config, embedder, mode='test',
                                    path=args.input, embedder_path=embedder_path)

    model = Seq2Seq(config, embedder=data_loader.embedder)
    model.load_state_dict(checkpoint['state_dict'])
    if not args.no_cuda:
        model.cuda()
    model.eval()
    model.summary()

    result = []
    from tqdm import tqdm
    for batch_idx, in_seq in enumerate(tqdm(data_loader)):
    # for batch_idx, in_seq in enumerate(data_loader):
        in_seq = torch.FloatTensor(in_seq)
        in_seq = Variable(in_seq)
        if not args.no_cuda:
            in_seq = in_seq.cuda()
        if args.beam_size == 1:
            out_seq = model(in_seq, 24)
            out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
            out_seq = np.transpose(out_seq, (1, 0, 2))
            out_seq = data_loader.embedder.decode_lines(out_seq)
        else:
            out_seq = beam_search(model, data_loader.embedder, in_seq, seq_len=24, beam_size=args.beam_size)
            out_seq = data_loader.embedder.decode_lines(out_seq)

        out_seq = [postprocess(line) for line in out_seq]
        result.extend(out_seq)

    with open(args.output, 'w') as f:
        for line in result:
            f.write(line+'\n')


def postprocess(raw):
    line = re.sub(r'<UNK>', '', raw)
    line = re.sub(r'\.{4,}', '...', line)
    line = re.sub(r'了+', '了', line)
    line = re.sub(r'[a-zA-Z0-9]', '', line)
    line = re.sub(r'\^', '', line)
    line = re.sub(r'((.)\2+)', r'\g<2>', line)
    if len(line) == 0:
        line = '你好'
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2-2 Testing')
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
    parser.add_argument('--beam-size', default=1, type=int,
                        help='beam search size n (default: 1)')

    args = parser.parse_args()
    main(args)
