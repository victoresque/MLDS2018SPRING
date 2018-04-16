import argparse
import logging
import torch.optim as optim
from model.seq2seq import Seq2Seq
from model.loss import cross_entropy
from model.metric import bleu
from data_loader import CaptionDataLoader
from trainer import CaptionTrainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')

# TODO: (important) config file support

# FIXME: loss function correctness
# FIXME: training data (fixed label or random label)
# FIXME: more information in checkpoints
# FIXME: save state_dict or the full model?
# FIXME: (optional) caption_data_loader shouldn't return 3 objects since it'll affect training process

# TODO: (important) make sure the code is flexible enough to fit 2-1 and 2-2
# TODO: (important) implement TODOs in seq2seq.py
# TODO: control Seq2Seq parameters from arguments
# TODO: create a base class for all embedders
# TODO: check code clarity and readability
# TODO: word embedding
# TODO: make sure different enhancements of Seq2Seq can be toggled

# DONE: folder structure (embedding.py -> preprocess/embedding.py)

# NOTE: coding style should follow PEP8


parser = argparse.ArgumentParser(description='HW2 Training')
parser.add_argument('-c', '--config', default='', type=str,
                    help='config file path (default: none)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('-e', '--epochs', default=1000, type=int,
                    help='number of total epochs (default: 1000)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved/', type=str,
                    help='directory of saved model (default: saved/)')
parser.add_argument('--save-freq', default=50, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.0)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')
# HW2 specific arguments
parser.add_argument('--task', required=True, type=str,
                    help='Specify the task to train [caption, chatbot]')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate (default: 1e-4)')
parser.add_argument('--rnn-type', default='LSTM', type=str,
                    help='Seq2Seq RNN mode [LSTM, GRU] (default: LSTM)')
parser.add_argument('--hidden-size', default=8, type=int,
                    help='Seq2Seq hidden features dimension (default: 256)')
parser.add_argument('--emb-size', default=32, type=int,
                    help='Word embedding dimension (default: 1000)')


def main(args):
    train_logger = Logger()
    training_name = args.task.title()
    if args.task.lower() == 'caption':
        model = Seq2Seq(hidden_size=args.hidden_size,
                        output_size=args.emb_size,
                        rnn_type=args.rnn_type)
        model.summary()

        data_loader = CaptionDataLoader(data_dir=args.data_dir,
                                        batch_size=args.batch_size,
                                        emb_size=args.emb_size,
                                        shuffle=True, mode='train')
        valid_data_loader = data_loader.split_validation(args.validation_split)

        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        loss = cross_entropy
        metrics = [bleu]
        trainer = CaptionTrainer(model, loss, metrics,
                                 data_loader=data_loader,
                                 valid_data_loader=valid_data_loader,
                                 optimizer=optimizer,
                                 epochs=args.epochs,
                                 train_logger=train_logger,
                                 save_dir=args.save_dir,
                                 save_freq=args.save_freq,
                                 resume=args.resume,
                                 verbosity=args.verbosity,
                                 training_name=training_name,
                                 with_cuda=not args.no_cuda,
                                 monitor='val_bleu',
                                 monitor_mode='max')
    else:
        model = None
        data_loader = None
        data_loader, valid_data_loader = None, None
        optimizer = None
        loss = None
        metrics = []
        trainer = None

    trainer.train()


if __name__ == '__main__':
    main(parser.parse_args())
