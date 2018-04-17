import argparse
import logging
import torch.optim as optim
from model.seq2seq import Seq2Seq
from model.loss import cross_entropy, mse_loss
from model.metric import bleu
from data_loader import CaptionDataLoader, ChatbotDataLoader
from trainer import CaptionTrainer
from preprocess.embedding import OneHotEmbedder, Word2VecEmbedder
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')

# TODO: (important) trainer config file support
# TODO: (important) model config file support
# TODO: (important) save configs in checkpoints
# TODO: change those fucking arguments to *args, **kwargs

# FIXME: loss function correctness
# FIXME: training data (fixed label or random label)
# FIXME: more information in checkpoints

# TODO: (important) make sure the code is flexible enough to fit 2-1 and 2-2
# TODO: (important) implement TODOs in seq2seq.py
# TODO: control Seq2Seq parameters from arguments
# TODO: check code clarity and readability
# TODO: make sure different enhancements of Seq2Seq can be toggled

# NOTE: coding style should follow PEP8


parser = argparse.ArgumentParser(description='HW2 Training')
parser.add_argument('-c', '--config', default='', type=str,
                    help='config file path (default: none)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('-e', '--epochs', default=1000, type=int,
                    help='number of total epochs (default: 1000)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved/', type=str,
                    help='directory of saved model (default: saved/)')
parser.add_argument('--save-freq', default=5, type=int,
                    help='training checkpoint frequency (default: 5)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.0)')
parser.add_argument('--name', required=True, type=str,
                    help='training session name')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')
# HW2 specific arguments
parser.add_argument('--task', required=True, type=str,
                    help='Specify the task to train [caption, chatbot]')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate (default: 1e-3)')
parser.add_argument('--rnn-type', default='LSTM', type=str,
                    help='Seq2Seq RNN mode [LSTM, GRU] (default: LSTM)')
parser.add_argument('--hidden-size', default=256, type=int,
                    help='Seq2Seq hidden features dimension (default: 256)')
parser.add_argument('--emb-size', default=1024, type=int,
                    help='Word embedding dimension (default: 1024)')


def main(args):
    # print(vars(args))
    train_logger = Logger()
    training_name = args.name
    if args.task.lower() == 'caption':
        embedder = OneHotEmbedder
        # embedder = Word2VecEmbedder
        data_loader = CaptionDataLoader(data_dir=args.data_dir,
                                        batch_size=args.batch_size,
                                        embedder=embedder,
                                        emb_size=args.emb_size,
                                        shuffle=True, mode='train')
        valid_data_loader = data_loader.split_validation(args.validation_split)

        model = Seq2Seq(input_size=4096,
                        hidden_size=args.hidden_size,
                        output_size=args.emb_size,
                        embedder=data_loader.embedder,
                        rnn_type=args.rnn_type)
        model.summary()

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
        # TODO
        embedder = OneHotEmbedder
        data_loader = ChatbotDataLoader(data_dir=args.data_dir,
                                        batch_size=args.batch_size,
                                        embedder=embedder,
                                        emb_size=args.emb_size,
                                        shuffle=True, mode='train')
        valid_data_loader = data_loader.split_validation(args.validation_split)

        model = Seq2Seq(input_size=4096,
                        hidden_size=args.hidden_size,
                        output_size=args.emb_size,
                        embedder=data_loader.embedder,
                        rnn_type=args.rnn_type)

        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        loss = None
        metrics = []
        trainer = None

    trainer.train()


if __name__ == '__main__':
    main(parser.parse_args())
