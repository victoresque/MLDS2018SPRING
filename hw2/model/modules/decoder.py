import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, mode='GRU'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        if mode.upper() == 'GRU':
            self.rnn = nn.GRU
        elif mode.upper() == 'LSTM':
            self.rnn = nn.LSTM
        else:
            raise Exception('Unknown cell type: {}'.format(mode))
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = self.rnn(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, in_seq, hidden):
        output = self.embedding(in_seq).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
