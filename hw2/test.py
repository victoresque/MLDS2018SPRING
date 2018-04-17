import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.seq2seq import Seq2Seq
from data_loader.caption_data_loader import CaptionDataLoader
from preprocess.embedding import OneHotEmbedder, Word2VecEmbedder
from utils.util import ensure_dir


def main(args):
    checkpoint = torch.load(args.checkpoint)
    config = checkpoint['config']

    if args.task.lower() == 'caption':
        embedder = eval(config['embedder'])
        data_loader = CaptionDataLoader(data_dir=args.data_dir,
                                        batch_size=1,
                                        embedder=embedder,
                                        emb_size=config['data_loader']['emb_size'],
                                        mode='test', shuffle=False)

        model = Seq2Seq(config['model'], embedder=data_loader.embedder)
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
            out_seq = data_loader.embedder.decode_lines(out_seq)
            out_seq = [(fmt[j]['id'], line) for j, line in enumerate(out_seq)]
            result.extend(out_seq)

        ensure_dir('results')
        with open('results/prediction.txt', 'w') as f:
            for video_id, caption in result:
                caption = postprocess(caption)
                f.write(video_id+','+caption+'\n')
    else:
        pass


def postprocess(raw):
    raw = raw.split()
    line = []
    prev_word = None
    raw = [w for w in raw if w != '<UNK>']
    for word in raw:
        if word != prev_word:
            line.append(word)
            prev_word = word
    return ' '.join(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2 Testing')
    parser.add_argument('--task', required=True, type=str,
                        help='Specify the task to train [caption, chatbot]')
    parser.add_argument('--data-dir', default='datasets', type=str,
                        help='directory of training/testing data (default: datasets)')
    parser.add_argument('--no-cuda', action="store_true",
                        help='use CPU instead of GPU')
    # HW2 specific arguments
    parser.add_argument('--checkpoint',
                        default='saved/Caption_seq2seq_onehot_256_1024/'
                                'checkpoint-epoch080-bleu-0.84-val_bleu-0.70.pth.tar',
                        type=str, help='model path')
    parser.add_argument('--source', default='dataset', type=str,
                        help='source [dataset, file] (default: dataset)')

    args = parser.parse_args()
    main(args)
