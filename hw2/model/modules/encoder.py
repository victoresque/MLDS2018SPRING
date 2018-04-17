import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_size = config['hidden_size']
        self.rnn_type = config['rnn_type'].upper()
        self.rnn = eval('nn.' + self.rnn_type)(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['layers'],
            dropout=config['enc_dropout'],
            bidirectional=False
        )

    def forward(self, in_seq):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        output, hidden = self.rnn(in_seq)
        return output, hidden
