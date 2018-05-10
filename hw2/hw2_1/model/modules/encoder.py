import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_size = config['model']['hidden_size']
        self.rnn_type = config['model']['rnn_type'].upper()
        self.rnn = eval('nn.' + self.rnn_type)(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['encoder']['layers'],
            dropout=config['model']['encoder']['dropout'],
            bidirectional=config['model']['encoder']['bidirectional']
        )

    def forward(self, in_seq):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        output, hidden = self.rnn(in_seq)
        return output, hidden
