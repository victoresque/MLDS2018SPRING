import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, mode='GRU'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        if mode.upper() == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True,
                              num_layers=1, dropout=0, bidirectional=False)
        elif mode.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True,
                               num_layers=1, dropout=0, bidirectional=False)
        else:
            raise Exception('Unknown cell type: {}'.format(mode))

    def forward(self, in_seq):
        output, hc = self.rnn(in_seq)
        return output, hc

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
