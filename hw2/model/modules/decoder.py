from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, mode='GRU', max_length=16, tokens=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mode = mode
        if mode.upper() == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=False,
                              num_layers=2, dropout=0., bidirectional=False)
        elif mode.upper() == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=False,
                               num_layers=2, dropout=0., bidirectional=False)
        else:
            raise Exception('Unknown cell type: {}'.format(mode))
        self.emb_in = nn.Linear(output_size, hidden_size)
        self.emb_out = nn.Linear(hidden_size, output_size)
        self.max_length = max_length
        self.tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3} \
            if tokens is None else tokens

    def forward(self, hidden):
        out_seq = []

        def onehot(dim, label):
            v = np.zeros((dim,))
            v[label] = 1
            return v

        bos_onehot = onehot(self.output_size, self.tokens['<BOS>'])
        if self.mode == 'GRU':
            dec_in = Variable(torch.FloatTensor([[bos_onehot for _ in range(hidden.data.shape[1])]]))
        else:
            dec_in = Variable(torch.FloatTensor([[bos_onehot for _ in range(hidden[0].data.shape[1])]]))
        if next(self.parameters()).is_cuda:
            dec_in = dec_in.cuda()
        for i in range(self.max_length):
            dec_in = self.emb_in(dec_in)
            if self.mode == 'GRU':
                dec_out, hidden = self.rnn(dec_in, hidden)
            else:
                dec_out, (hidden, c) = self.rnn(dec_in, hidden)
                hidden = (hidden, c)
            dec_out = self.emb_out(dec_out)
            out_seq.append(dec_out[0])
            dec_in = copy(F.softmax(dec_out, 1))

        return out_seq
