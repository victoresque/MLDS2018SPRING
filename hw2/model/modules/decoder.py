import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, cell_type='GRU'):
        super(Decoder, self).__init__()
        if cell_type.upper() == 'GRU':
            self.cell = nn.GRUCell
        elif cell_type.upper() == 'LSTM':
            self.cell = nn.LSTMCell
        else:
            raise Exception('Unknown cell type: {}'.format(cell_type))

    def forward(self, x):
        pass
