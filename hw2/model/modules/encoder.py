import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, mode='GRU'):
        super(Encoder, self).__init__()
        if mode.upper() == 'GRU':
            self.rnn = nn.GRU
        elif mode.upper() == 'LSTM':
            self.rnn = nn.LSTM
        else:
            raise Exception('Unknown cell type: {}'.format(mode))

    def forward(self, x):
        pass
