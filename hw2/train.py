import argparse
import torch.optim as optim
from model.seq2seq import Seq2Seq
from model.loss import cross_entropy
from model.metric import bleu
from data_loader.caption_data_loader import CaptionDataLoader
from trainer.caption_trainer import CaptionTrainer
from logger.logger import Logger

# FIXME: loss function correctness
# FIXME: training data (fixed label or random label)
# FIXME: more information in checkpoint
# FIXME: save state_dict or the full model?
# FIXME: (optional) caption_data_loader shouldn't return 3 objects since it'll affect training process
# TODO: (important) make sure the code is flexible enough to fit 2-1 and 2-2
# TODO: (important) implement TODOs in seq2seq.py
# TODO: control Seq2Seq parameters from arguments
# TODO: create a base class for all embedders
# TODO: folder structure (embedding.py -> preprocess/embedding.py)
# TODO: check code clarity and readability
# TODO: word embedding
# TODO: make sure different enhancements of Seq2Seq can be toggled
# NOTE: coding style should follow PEP8


parser = argparse.ArgumentParser(description='HW2 Training')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('-e', '--epochs', default=1000, type=int,
                    help='number of total epochs (default: 1000)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=1, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 1)')
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


def main(args):
    logger = Logger()
    identifier = args.task.title() + '_'
    if args.task.lower() == 'caption':
        model = Seq2Seq()

        data_loader = CaptionDataLoader(args.data_dir, args.batch_size)
        valid_data_loader = data_loader.split_validation(args.validation_split)

        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
        loss = cross_entropy
        metrics = [bleu]
        trainer = CaptionTrainer(model, loss, metrics,
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
    else:
        model = None
        data_loader = None
        data_loader, valid_data_loader = None, None
        optimizer = None
        loss = None
        metrics = []
        trainer = None

    model.summary()
    trainer.train()


if __name__ == '__main__':
    main(parser.parse_args())
