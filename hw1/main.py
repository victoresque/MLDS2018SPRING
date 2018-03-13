import argparse
import torch.optim as optim

from models.models import DeepFC, MiddleFC, ShallowFC
from models.models import DeepMnistCNN, MiddleMnistCNN, ShallowMnistCNN
from models.models import DeepCifarCNN, MiddleCifarCNN, ShallowCifarCNN
from models.loss import mse_loss, cross_entropy_loss
from models.metric import accuracy
from data_loader.function_data_loader import FunctionDataLoader
from data_loader.data_loader import MnistLoader, CifarLoader
from trainers.trainer import Trainer
from logger.logger import Logger

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=20, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-v', '--verbosity', default=1, type=int,
                    help='verbosity [0: quiet, 1: per epoch, 2: complete] (default: 2)')
parser.add_argument('--save-dir', default='models/saved', type=str,
                    help='directory of saved model (default: models/saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 5)')
parser.add_argument('--data-dir', default='data/datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU in case there\'s no GPU support')
# HW1 specific arguments
parser.add_argument('--dataset', default='func', type=str,
                    help='training data [mnist, cifar, func] (default: func)')
parser.add_argument('--target-func', default='sinc', type=str,
                    help='target function [sin, sinc, ceil, damp] (default: sin)')
parser.add_argument('--arch', default='deep', type=str,
                    help='model architecture [deep, middle, shallow] (default: deep)')
'''  HW1 requirements
  1. function regression
     (1) >=3 models
     (2) >=2 functions
     (3) plot the function (ground truth + predicted)
         [total: >=6 charts]
     (4) plot a loss-epoch chart for each model
         [total: >=6 charts]
  2. real problem
     (1) >=3 models
     (2) >=2 tasks
     (3) plot a loss-epoch chart for each model
         [total: >=6 charts]
     (4) plot a accuracy-epoch chart for each model
         [total: >=6 charts]
'''


def main(args):
    logger = Logger()
    if args.dataset != 'func':
        loss = cross_entropy_loss
        metrics = [accuracy]
        model = eval(args.arch.title() + args.dataset.title() + 'CNN')()
        data_loader = eval(args.dataset.title() + 'Loader')(args.batch_size)
    else:
        loss = mse_loss
        metrics = []
        model = eval(args.arch.title() + 'FC')()
        data_loader = FunctionDataLoader(args.target_func,
                                         batch_size=args.batch_size,
                                         n_sample=32768, x_range=(0, 1))

    model.summary()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, data_loader, loss, metrics,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      logger=logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      with_cuda=not args.no_cuda)
    trainer.train()
    # logger.print()
    print(logger.entries)
    

if __name__ == '__main__':
    main(parser.parse_args())
