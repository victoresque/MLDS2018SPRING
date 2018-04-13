import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, mode='GRU'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        if mode.upper() == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True,
                              num_layers=1, dropout=0, bidirectional=False)
        elif mode.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True,
                               num_layers=1, dropout=0, bidirectional=False)
        else:
            raise Exception('Unknown cell type: {}'.format(mode))

    def forward(self, in_seq):
        output, hidden = self.rnn(in_seq)
        return output, hidden
