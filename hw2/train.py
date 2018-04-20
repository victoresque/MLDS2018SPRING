import os
import logging
import argparse
from model.seq2seq import Seq2Seq
from model.loss import *
from model.metric import *
from data_loader import *
from trainer import CaptionTrainer
from preprocess.embedding import *
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')

# TODO: (important) HW2-2
# TODO: (important) implement TODOs in seq2seq.py
# TODO: specify Word2VecEmbedder model path
# NOTE: currently training target is randomly picked
# NOTE: coding style should follow PEP8


def main(config, resume):
    train_logger = Logger()

    embedder = eval(config['embedder']['type'])
    data_loader = CaptionDataLoader(config, embedder, mode='train')
    valid_data_loader = data_loader.split_validation()

    model = eval(config['arch'])(config, data_loader.embedder)
    model.summary()

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = CaptionTrainer(model, loss, metrics,
                             resume=resume,
                             config=config,
                             data_loader=data_loader,
                             valid_data_loader=valid_data_loader,
                             train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if os.path.exists(path):
            opt = input('Warning: path {} already exists, continue? [y/N] '.format(path))
            if opt.upper() != 'Y':
                exit(1)
    assert config is not None

    main(config, args.resume)
