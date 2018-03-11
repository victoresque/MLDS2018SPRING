import argparse
from models.model import Model
from data.data_loader import DataLoader
from trainers.trainer import Trainer
import torch.optim as optim

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=32, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='models/saved', type=str,
                    help='directory of saved model (default: models/saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 5)')
parser.add_argument('--data-dir', default='data/datasets', type=str,
                    help='directory of training/testing data (default: datasets)')


def main(args):
    model = Model()
    model.summary()
    optimizer = optim.Adam(model.parameters())
    data_loader = DataLoader(args.data_dir)
    trainer = Trainer(model,
                      data_loader,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    main(parser.parse_args())
