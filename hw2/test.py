import argparse
import torch
from torch.autograd import Variable
import numpy as np
from model.seq2seq import Seq2Seq
from data_loader.caption_data_loader import CaptionDataLoader
from utils.util import ensure_dir


parser = argparse.ArgumentParser(description='HW2 Testing')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')
# HW2 specific arguments
parser.add_argument('--model', default='saved/CaptionW2V/checkpoint-epoch2200-bleu-0.80-val_bleu-0.57.pth.tar', type=str,
                    help='model path')
parser.add_argument('--task', required=True, type=str,
                    help='Specify the task to train [caption, chatbot]')
parser.add_argument('--source', default='dataset', type=str,
                    help='source [dataset, file] (default: dataset)')


def main(args):
    if args.task.lower() == 'caption':
        model = Seq2Seq(hidden_size=256, output_size=256, rnn_type='LSTM')
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
        model.summary()
        if not args.no_cuda:
            model.cuda()
        model.eval()
        data_loader = CaptionDataLoader(args.data_dir,
                                        batch_size=1,
                                        emb_size=256,
                                        mode='test')

        result = []
        for batch_idx, (in_seq, fmt) in enumerate(data_loader):
            in_seq = torch.FloatTensor(in_seq)
            in_seq = Variable(in_seq)
            if not args.no_cuda:
                in_seq = in_seq.cuda()
            out_seq = model(in_seq)
            out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
            out_seq = np.transpose(out_seq, (1, 0, 2))
            out_seq = data_loader.embedder.decode_lines(out_seq)
            out_seq = [(fmt[j]['id'], line) for j, line in enumerate(out_seq)]
            result.extend(out_seq)

        ensure_dir('results')
        with open('results/prediction.txt', 'w') as f:
            for video_id, caption in result:
                f.write(video_id+','+caption+'\n')
    else:
        pass


if __name__ == '__main__':
    main(parser.parse_args())
