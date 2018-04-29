import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel
from .modules import Encoder, Decoder, DecoderAttn
import numpy as np

class Seq2Seq(BaseModel):
    """
    Note:
        input:
            type:  Variable
            shape: random length in sample_range x batch size x 4096
        output:
            type:  Variable
            shape: max sequence length in batch x batch size x emb size
    """
    def __init__(self, config, embedder):
        super(Seq2Seq, self).__init__(config)
        self.input_size = config['model']['input_size']
        self.embedder = embedder
        self.rnn_type = config['model']['rnn_type'].upper()
        self.encoder = Encoder(config)
        self.attn = config['model']['attention']
        if self.attn:
            self.decoder = DecoderAttn(config, self.embedder)
            self.z_0 = nn.Parameter(torch.rand(1, config['model']['hidden_size']), requires_grad=True)
        else:
            self.decoder = Decoder(config, self.embedder)
    
    # epoch for schedule sampling with respect ot time
    # targ_idx for bias distribution over time
    # targ_seq for teacher forcing / scheduled sampling
    def forward(self, in_seq, seq_len, targ_seq=None, targ_idx=None, epoch=None):
        if targ_idx is None: targ_idx = np.array([seq_len//2])
        enc_out, hidden = self.encoder(in_seq)
        
        bos = self.embedder.encode_word('<BOS>')
        n_batch = hidden[0].data.shape[1] if self.rnn_type == 'LSTM' else hidden.data.shape[1]
        dec_in = Variable(torch.FloatTensor(np.array([[bos for _ in range(n_batch)]])))
        
        if self.attn:
            hiddens = (hidden, hidden)
            z_batch = self.z_0.repeat(enc_out.size(1), 1) #(batch, hidden)
            (out_seq, _), _ = self.decoder(enc_out, dec_in, z_batch, hiddens, seq_len, targ_idx, epoch, targ_seq)
        else:
            out_seq, _ = self.decoder(enc_out, dec_in, hidden, seq_len, epoch, targ_seq)
            
        return out_seq
