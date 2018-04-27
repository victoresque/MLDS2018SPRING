from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, config, embedder):
        super(Decoder, self).__init__()
        self.hidden_size = config['model']['hidden_size']
        self.embedder = embedder
        self.output_size = embedder.emb_size
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

    def forward(self, enc_out, hidden, seq_len, targ_seq=None):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        out_seq = []
        bos = self.embedder.encode_word('<BOS>')
        n_batch = hidden[0].data.shape[1] if self.rnn_type == 'LSTM' else hidden.data.shape[1]
        dec_in = Variable(torch.FloatTensor(np.array([[bos for _ in range(n_batch)]])))
        with_cuda = next(self.parameters()).is_cuda

        for i in range(seq_len):
            dec_in = dec_in.cuda() if with_cuda else dec_in
            dec_in = self.emb_in(dec_in)
            dec_out, hidden = self.rnn(dec_in, hidden)
            dec_out = self.emb_out(dec_out)
            out_seq.append(dec_out[0])
            dec_out = F.tanh(dec_out)
            if self.training and np.random.rand() > self.scheduled_sampling:
                dec_in = targ_seq[i:i+1]
            else:
                dec_in = self.embedder.dec_out2dec_in(dec_out)
                dec_in = Variable(torch.FloatTensor(dec_in))

        out_seq = torch.stack(out_seq)
        return out_seq


class DecoderAttn(nn.Module):
    def __init__(self, config, embedder):
        super(DecoderAttn, self).__init__()
        self.hidden_size = config['model']['hidden_size']
        self.embedder = embedder
        self.output_size = embedder.emb_size
        self.rnn_type = config['model']['rnn_type'].upper()
        self.rnn = eval('nn.' + self.rnn_type)(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config['model']['decoder']['layers'],
            dropout=config['model']['decoder']['dropout'],
            bidirectional=False
        )
        self.scheduled_sampling = config['model']['scheduled_sampling']
        self.emb_in = nn.Linear(self.output_size+self.hidden_size, self.hidden_size)
        self.emb_out = nn.Linear(self.hidden_size, self.output_size)
        
        self.keyrnn = eval('nn.' + self.rnn_type)(
            input_size=2*self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config['model']['decoder']['layers'],
            dropout=config['model']['decoder']['attnkey_dropout'],
            bidirectional=False
        )
        self.z_0 = nn.Parameter(torch.rand(1, 1, config['model']['hidden_size']), requires_grad=True)

    def forward(self, enc_out, hidden, seq_len, targ_seq=None):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        out_seq = []
        bos = self.embedder.encode_word('<BOS>')
        n_batch = hidden[0].data.shape[1] if self.rnn_type == 'LSTM' else hidden.data.shape[1]
        dec_in = Variable(torch.FloatTensor(np.array([[bos for _ in range(n_batch)]])))
        with_cuda = next(self.parameters()).is_cuda
        
        z_batch = self.z_0.repeat(1, enc_out.size(1), 1)
        key_hidden = hidden

        for i in range(seq_len):
            z_seq = z_batch.repeat(enc_out.size(0), 1, 1)
            attn_weight = F.cosine_similarity(z_seq, enc_out, dim=2)
            # attn_weight = F.softmax(attn_weight, dim=0)
            attn_weight = attn_weight.view(*attn_weight.size(),1)
            attn_weight_repeat = attn_weight.repeat(1,1,self.hidden_size)
            attn_out = attn_weight_repeat.mul(enc_out)
            attn_out = attn_out.sum(dim=0)
            attn_out = attn_out.view(1, *attn_out.size())
            
            keygen_in = torch.cat((attn_out, z_batch), dim=2)
            z_batch, key_hidden = self.keyrnn(keygen_in, key_hidden)
            
            dec_in = dec_in.cuda() if with_cuda else dec_in
            dec_in = torch.cat((attn_out, dec_in), dim=2)
            dec_in = self.emb_in(dec_in)
            dec_out, hidden = self.rnn(dec_in, hidden)
            dec_out = self.emb_out(dec_out)
            out_seq.append(dec_out[0])
            if self.training and np.random.rand() > self.scheduled_sampling:
                dec_in = targ_seq[i:i+1]
            else:
                dec_in = self.embedder.dec_out2dec_in(dec_out)
                dec_in = Variable(torch.FloatTensor(dec_in))

        out_seq = torch.stack(out_seq)
        return out_seq

