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

# FIXME: loss function correctness
# FIXME: training data (fixed label or random label)
# FIXME: more information in checkpoints

# TODO: (important) make sure the code is flexible enough to fit 2-1 and 2-2
# TODO: (important) implement TODOs in seq2seq.py
# TODO: control Seq2Seq parameters from arguments
# TODO: check code clarity and readability
# TODO: make sure different enhancements of Seq2Seq can be toggled

# NOTE: coding style should follow PEP8


def main(config, resume):
    train_logger = Logger()
    if config['task'] == 'caption':
        embedder = eval(config['embedder'])
        data_loader = CaptionDataLoader(embedder=embedder,
                                        mode='train',
                                        **config['data_loader'])
        valid_data_loader = data_loader.split_validation(config['validation_split'])

        model = Seq2Seq(config['model'], embedder=data_loader.embedder)
        model.summary()

        optimizer = eval('optim.'+config['optimizer_type'])(model.parameters(), **config['optimizer'])
        loss = eval(config['loss'])
        metrics = [bleu]

        trainer = CaptionTrainer(model, loss, metrics,
                                 data_loader=data_loader,
                                 valid_data_loader=valid_data_loader,
                                 optimizer=optimizer,
                                 train_logger=train_logger,
                                 training_name=config['name'],
                                 with_cuda=config['cuda'],
                                 resume=resume,
                                 config=config,
                                 **config['trainer'])
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
    parser = argparse.ArgumentParser(description='HW2 Training')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--force', action='store_true',
                        help='Ignore existing folder')

    args = parser.parse_args()
    assert (args.config is None) != (args.resume is None)
    if args.resume is not None:
        import torch
        config = torch.load(args.resume)['config']
    else:
        import json
        config = json.load(open(args.config))

    if not args.force:
        import os
        assert not os.path.exists(os.path.join(config['trainer']['save_dir'], config['name']))

    main(config, args.resume is not None)
