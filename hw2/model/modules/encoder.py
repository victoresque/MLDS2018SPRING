import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.upper()
        self.rnn = eval('nn.' + self.rnn_type)(
            input_size, hidden_size, batch_first=False,
            num_layers=2, dropout=0., bidirectional=False
        )

    def forward(self, in_seq):
        output, hidden = self.rnn(in_seq)
        return output, hidden
