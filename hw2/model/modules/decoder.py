from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.hidden_size = config['model']['hidden_size']
        self.output_size = config['embedder']['emb_size']
        self.rnn_type = config['model']['rnn_type'].upper()
        self.rnn = eval('nn.' + self.rnn_type)(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config['model']['decoder']['layers'],
            dropout=config['model']['decoder']['dropout'],
            bidirectional=False
        )
        self.scheduled_sampling = config['model']['scheduled_sampling']
        self.emb_in = nn.Linear(self.output_size, self.hidden_size)
        self.emb_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, enc_out, hidden, seq_len, embedder, targ_seq=None):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        out_seq = []
        bos = embedder.encode_word('<BOS>')
        n_batch = hidden[0].data.shape[1] if self.rnn_type == 'LSTM' else hidden.data.shape[1]
        dec_in = Variable(torch.FloatTensor(np.array([[bos for _ in range(n_batch)]])))
        with_cuda = next(self.parameters()).is_cuda

        for i in range(seq_len):
            dec_in = dec_in.cuda() if with_cuda else dec_in
            dec_in = self.emb_in(dec_in)
            dec_out, hidden = self.rnn(dec_in, hidden)
            dec_out = self.emb_out(dec_out)
            out_seq.append(dec_out[0])
            if self.training and np.random.rand() > self.scheduled_sampling:
                dec_in = targ_seq[i:i+1]
            else:
                dec_in = embedder.dec_out2dec_in(dec_out)
                dec_in = Variable(torch.FloatTensor(dec_in))

        out_seq = torch.stack(out_seq)
        return out_seq
