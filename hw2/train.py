import argparse
import torch.optim as optim
from model.seq2seq import Seq2Seq
from model.loss import cross_entropy
from model.metric import bleu
from data_loader.caption_data_loader import CaptionDataLoader
from utils.util import split_validation
from trainer.trainer import Trainer
from logger.logger import Logger

parser = argparse.ArgumentParser(description='HW2 Training')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=200, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=1, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved/', type=str,
                    help='directory of saved model (default: saved/)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.0)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')
# HW2 specific arguments
parser.add_argument('--task', required=True,
                    help='Specify the task to train [caption, chatbot]')


def main(args):
    if args.task.lower() == 'caption':
        model = Seq2Seq
        loss = cross_entropy
        data_loader = CaptionDataLoader(args.data_dir, args.batch_size)
        data_loader, valid_data_loader = split_validation(data_loader, args.validation_split)
        metrics = [bleu]
    else:
        model = None
        loss = None
        data_loader = None
        data_loader, valid_data_loader = split_validation(data_loader, args.validation_split)
        metrics = []

    model.summary()
    logger = Logger()
    optimizer = optim.RMSprop(model.parameters())

    identifier = args.task.title() + '_'
    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      logger=logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      identifier=identifier,
                      with_cuda=not args.no_cuda)

    trainer.train()


if __name__ == '__main__':
    main(parser.parse_args())
