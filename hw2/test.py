import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.seq2seq import Seq2Seq
from data_loader import *
from preprocess.embedding import *
from utils.util import ensure_dir
from utils.beam_search import beam_search


def main(args):
    checkpoint = torch.load(args.checkpoint)
    config = checkpoint['config']

    if args.task.lower() == 'caption':
        embedder = eval(config['embedder']['type'])
        data_loader = CaptionDataLoader(config, embedder, mode='test',
                                        path=args.data_dir, embedder_path=args.embedder_path)

        model = Seq2Seq(config, embedder=data_loader.embedder)
        model.load_state_dict(checkpoint['state_dict'])
        if not args.no_cuda:
            model.cuda()
        model.eval()
        model.summary()

        result = []
        for batch_idx, (in_seq, fmt) in enumerate(data_loader):
            in_seq = torch.FloatTensor(in_seq)
            in_seq = Variable(in_seq)
            if not args.no_cuda:
                in_seq = in_seq.cuda()
            out_seq = model(in_seq, 24)
            out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
            out_seq = np.transpose(out_seq, (1, 0, 2))
            if args.beam_size == 1:
                out_seq = data_loader.embedder.decode_lines(out_seq)
            else:
                seqs = beam_search(out_seq, args.beam_size)
                out_seq = []
                for seq in seqs:
                    line = [data_loader.embedder.word_list[int(word_idx)] for word_idx in list(seq)]
                    line = ' '.join(line)
                    line = line.split('<EOS>', 1)[0]
                    line = line.split('<PAD>', 1)[0]
                    line = line.split()
                    if len(line) == 0:
                        line = ['a']
                    line = ' '.join(line)
                    out_seq.append(line)

            out_seq = [(fmt[0]['id'], out_seq)]
            result.extend(out_seq)

        ensure_dir('results')
        with open(args.output, 'w') as f:
            for video_id, caption in result:
                caption = postprocess(caption)
                f.write(video_id+','+caption+'\n')
    else:
        pass


def postprocess(raw):
    lines = []
    for seq in raw:
        seq = seq.split()
        prev_word = None
        seq = [w for w in seq if w != '<UNK>']
        line = []
        for word in seq:
            if word != prev_word:
                line.append(word)
                prev_word = word
        line = ' '.join(line)
        lines.append(line)
    return ','.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2 Testing')
    parser.add_argument('--task', required=True, type=str,
                        help='Specify the task to train [caption, chatbot]')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='model path')
    parser.add_argument('--data-dir', required=True, type=str,
                        help='input data directory')
    parser.add_argument('--embedder-path', required=True, type=str,
                        help='path of saved embedder')
    parser.add_argument('--output', required=True, type=str,
                        help='output filename')
    parser.add_argument('--no-cuda', action="store_true",
                        help='use CPU instead of GPU')
    parser.add_argument('--beam-size', default=1, type=int,
                        help='beam search size n (default: 1)')

    args = parser.parse_args()
    main(args)
