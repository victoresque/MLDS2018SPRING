import argparse
import torch.optim as optim
import torch.nn.functional as f
from models.model import Model
from data_loader.data_loader import DataLoader
from trainers.trainer import Trainer
from logger.logger import Logger

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='models/saved', type=str,
                    help='directory of saved model (default: models/saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 5)')
parser.add_argument('--data-dir', default='data/datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU in case there\'s no GPU support')


def main(args):
    model = Model()
    model.summary()
    loss = f.nll_loss
    optimizer = optim.Adam(model.parameters())
    data_loader = DataLoader(args.data_dir, args.batch_size)
    logger = Logger()
    trainer = Trainer(model, data_loader, loss,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      logger=logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      with_cuda=not args.no_cuda)
    trainer.train()


if __name__ == '__main__':
    main(parser.parse_args())
