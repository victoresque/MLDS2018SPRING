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


def main(args):
    checkpoint_path = os.path.join("saved/", args.name, args.checkpoint)
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    #if args.task.lower() == 'caption':
    embedder = eval(config['embedder']['type'])
    embedder_path = os.path.join("saved/", args.name, "embedder.pkl")
    data_loader = CaptionDataLoader(config, embedder, mode='test',
                                    path=args.data_dir, embedder_path=embedder_path)

    model = Seq2Seq(config, embedder=data_loader.embedder)
    model.load_state_dict(checkpoint['state_dict'])
    if not args.no_cuda:
        model.cuda()
    model.eval()
    model.summary()

    result = []
    for batch_idx, (in_seq, id) in enumerate(data_loader):
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

        out_seq = [(str(id[0]), out_seq)]
        result.extend(out_seq)

    output_path = os.path.join("datasets/MLDS_hw2_1_data/predict/", args.output)
    with open(output_path, 'w') as f:
        for video_id, caption in result:
            caption = postprocess(caption)
            f.write(video_id+','+caption+'\n')
    os.chdir("datasets/MLDS_hw2_1_data/")
    subprocess.call(["python3", "bleu_eval.py", "predict/"+args.output])


def postprocess(raw):
    lines = []
    for line in raw:
        line = line.split()
        line = [w.replace("<UNK>", "something") for w in line]
        for _ in range(3):
            line = remove_repeat(line)
            line = remove_duplicate(line)
            line = remove_dummyword(line)
        line = ' '.join(line)
        lines.append(line)
    return ','.join(lines)


def remove_duplicate(line):
    length = 2
    while length < len(line):
        stack = []
        skip_flag = False
        for i in range(len(line)+1-length):
            if skip_flag:
                skip_flag = False
                continue
            if (line[i], line[i+1]) not in stack:
                stack.append((line[i], line[i+1]))
            else:
                line[i], line[i+1] = None, None
                skip_flag = True
        line = [word for word in line if word is not None]
        length += 1
    return line

def remove_dummyword(line):
    stop_words = ['a','an','and','is','at','in','into','is','of','on','to','the','then', 'from', 'with']
    while line and line[-1] in stop_words:
        line.pop()
    return_line = []
    for i, word in enumerate(line):
        if word == 'something':
            if line[max(0,i-1)] == 'a' or line[max(0,i-1)] == 'an':
                return_line.pop()
        return_line.append(word)
    return return_line

def remove_repeat(seq):
    line = []
    prev_word = None
    for word in seq:
        if word != prev_word:
            line.append(word)
            prev_word = word
    return line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2-1 Testing')
    parser.add_argument('--name', required=True, type=str,
                        help='Specify the name of folder')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='model checkpoint file name')
    parser.add_argument('--data-dir', default='datasets/MLDS_hw2_1_data/testing_data/', type=str,
                        help='input data directory (default: datasets/MLDS_hw2_1_data/testing_data/)')
    parser.add_argument('--output', required=True, type=str,
                        help='output filename')
    parser.add_argument('--no-cuda', action="store_true",
                        help='use CPU instead of GPU')
    parser.add_argument('--beam-size', default=1, type=int,
                        help='beam search size n (default: 1)')

    args = parser.parse_args()
    main(args)
