from copy import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import poisson


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
        self.n_epochs = config['trainer']['epochs']
        self.emb_in = nn.Linear(self.output_size, self.hidden_size)
        self.emb_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, enc_out, dec_in, hidden, seq_len, epoch, targ_seq=None):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        out_seq = []
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
        return out_seq, hidden


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
        self.n_epochs = config['trainer']['epochs']
        self.emb_in = nn.Linear(self.output_size+self.hidden_size, self.hidden_size)
        self.emb_out = nn.Linear(self.hidden_size, self.output_size)
        
        self.keyrnn = eval('nn.' + self.rnn_type)(
            input_size=2*self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=config['model']['decoder']['layers'],
            dropout=config['model']['decoder']['attnkey_dropout'],
            bidirectional=False
        )
        self.key_out = nn.Linear(self.hidden_size, self.hidden_size)
        

    def forward(self, enc_out, dec_in, z_batch, hiddens, seq_len, targ_idx=None, epoch=None, targ_seq=None):
        """
        Note:
             input/return type/shape refer to seq2seq.py
        """
        out_seq = []
        with_cuda = next(self.parameters()).is_cuda
        
        # mu_array = targ_idx
        # bias_prob = [poisson.pmf(idx, mu_array) for idx in np.arange(enc_out.size(0))]
        # bias_prob = Variable(torch.FloatTensor(np.array(bias_prob))) #(seq_length, batch)
        # bias_prob = bias_prob.cuda() if with_cuda else bias_prob
        hidden_d, hidden_k = hiddens

        for i in range(seq_len):
            z_batch = self.key_out(z_batch) #(batch, hidden)
            z_batch = z_batch.view(1, *z_batch.size())
            z_seq = z_batch.repeat(enc_out.size(0), 1, 1)
            attn_weight = F.cosine_similarity(z_seq, enc_out, dim=2) #(seq_length, batch)
            # attn_weight = attn_weight.mul(bias_prob)
            attn_weight = F.softmax(attn_weight, dim=0)
            attn_weight = attn_weight.view(*attn_weight.size(),1) #(seq_length, batch, 1)
            attn_weight_repeat = attn_weight.repeat(1,1,self.hidden_size) #(seq_length, batch, hidden)
            attn_out = attn_weight_repeat.mul(enc_out) #(seq_length, batch, hidden)
            attn_out = attn_out.sum(dim=0) #(batch, hidden)
            attn_out = attn_out.view(1, *attn_out.size()) #(1, batch, hidden)
            
            keygen_in = torch.cat((attn_out, z_batch), dim=2) #(1, batch, 2*hidden)
            z_batch, hidden_k = self.keyrnn(keygen_in, hidden_k)
            z_batch = z_batch.view(z_batch.size(1), z_batch.size(2))
                        
            dec_in = dec_in.cuda() if with_cuda else dec_in
            dec_in = torch.cat((attn_out, dec_in), dim=2) #(1, batch, hidden+emb_size)
            dec_in = self.emb_in(dec_in) #(1, batch, hidden+emb_size)
            dec_out, hidden_d = self.rnn(dec_in, hidden_d)
            dec_out = self.emb_out(dec_out)
            out_seq.append(dec_out[0])
            
            if self.training and np.random.rand() > max(1e-8, 1-(epoch/100)):            # linear decay
            #if self.training and np.random.rand() > max(1e-8, 0.98**epoch):              # exponential decay
            #if self.training and np.random.rand() > max(1e-8, 10/(10+np.exp(epoch/10))): # inverse sigmoid decay
            #if self.training and np.random.rand() > self.scheduled_sampling:             # constant
                dec_in = targ_seq[i:i+1]
            else:
                dec_in = self.embedder.dec_out2dec_in(dec_out)
                dec_in = Variable(torch.FloatTensor(dec_in))

        out_seq = torch.stack(out_seq)
        return (out_seq, z_batch), (hidden_d, hidden_k)

