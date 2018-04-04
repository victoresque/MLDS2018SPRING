import argparse
import torch.optim as optim
from models.models import DeepFC, MiddleFC, ShallowFC
from models.models import DeepMnistCNN, MiddleMnistCNN, ShallowMnistCNN, DeeperMnistCNN
from models.models import DeepCifarCNN, MiddleCifarCNN, ShallowCifarCNN, DeeperCifarCNN
from models.loss import mse_loss, cross_entropy_loss
from models.metric import accuracy
from data_loader.function_data_loader import FunctionDataLoader
from data_loader.data_loader import MnistLoader, CifarLoader
from utils.util import split_validation
from trainers.trainer import Trainer
from logger.logger import Logger

parser = argparse.ArgumentParser(description='Homework 1')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=20000, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=1, type=int,
                    help='verbosity [0: quiet, 1: per epoch, 2: complete] (default: 2)')
parser.add_argument('--save-dir', default='models/saved', type=str,
                    help='directory of saved model (default: models/saved)')
parser.add_argument('--save-freq', default=20, type=int,
                    help='training checkpoint frequency (default: 5)')
parser.add_argument('--data-dir', default='data/datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.0, type=float,
                    help='ratio of split validation data ([0.0, 1.0), default: 0.0)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU in case there\'s no GPU support')
# HW1 specific arguments
parser.add_argument('--dataset', default='func', type=str,
                    help='training data [mnist, cifar, func] (default: func)')
parser.add_argument('--target-func', default='sinc', type=str,
                    help='target function [sin, sinc, stair, damp, square] (default: sinc)')
parser.add_argument('--arch', default='deep', type=str,
                    help='model architecture [deep, middle, shallow] (default: deep)')
parser.add_argument('--save-grad', action="store_true",
                    help='saving average gradient norm for HW1-2')
parser.add_argument('--rand-label', action="store_true",
                    help='shuffle all labels for HW1-3')


def main(args):
    logger = Logger()
    if args.dataset != 'func':
        loss = cross_entropy_loss
        metrics = [accuracy]
        if args.arch[:6] == 'deeper':
            model = eval(args.arch[:6].title() + args.dataset.title() + 'CNN')(int(args.arch[6:]))
            identifier = type(model).__name__ + args.arch[6:] + '_' + args.dataset + '_'
        else:
            model = eval(args.arch.title() + args.dataset.title() + 'CNN')()
            identifier = type(model).__name__ + '_' + args.dataset + '_'
        data_loader = eval(args.dataset.title() + 'Loader')(args.batch_size,
                                                            args.rand_label)
    else:
        loss = mse_loss
        metrics = []
        model = eval(args.arch.title() + 'FC')()
        data_loader = FunctionDataLoader(args.target_func,
                                         batch_size=args.batch_size,
                                         n_sample=1024, x_range=(0, 1))
        identifier = type(model).__name__ + '_' + args.target_func + '_'

    model.summary()
    optimizer = optim.Adam(model.parameters())
    data_loader, valid_data_loader = split_validation(data_loader, args.validation_split)
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
                      with_cuda=not args.no_cuda,
                      save_grad=args.save_grad)
    trainer.train()
    print(logger)


if __name__ == '__main__':
    main(parser.parse_args())
