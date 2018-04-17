from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, rnn_type='LSTM', max_length=24):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type.upper()
        self.rnn = eval('nn.' + self.rnn_type)(
            hidden_size, hidden_size, batch_first=False,
            num_layers=2, dropout=0., bidirectional=False
        )
        self.emb_in = nn.Linear(output_size, hidden_size)
        self.emb_out = nn.Linear(hidden_size, output_size)
        self.max_length = max_length

    def forward(self, enc_out, hidden, embedder):
        out_seq = []
        bos = embedder.encode_word('<BOS>')
        n_batch = hidden[0].data.shape[1] if self.rnn_type == 'LSTM' else hidden.data.shape[1]
        dec_in = Variable(torch.FloatTensor([[bos for _ in range(n_batch)]]))
        with_cuda = next(self.parameters()).is_cuda

        for i in range(self.max_length):
            dec_in = dec_in.cuda() if with_cuda else dec_in
            dec_in = self.emb_in(dec_in)
            dec_out, hidden = self.rnn(dec_in, hidden)
            dec_out = self.emb_out(dec_out)
            out_seq.append(dec_out[0])
            dec_in = embedder.dec_out2dec_in(dec_out)
            dec_in = Variable(torch.FloatTensor(dec_in))

        return out_seq
