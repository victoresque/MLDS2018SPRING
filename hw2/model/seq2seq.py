import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel
from model.modules import Encoder, Decoder

# TODO: Attention


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
        self.hidden_size = config['model']['hidden_size']
        self.emb_size = config['embedder']['emb_size']
        self.embedder = embedder
        self.rnn_type = config['model']['rnn_type'].upper()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, in_seq, seq_len, targ_seq=None):
        enc_out, hidden = self.encoder(in_seq)
        out_seq = self.decoder(enc_out, hidden, seq_len, self.embedder, targ_seq)
        return out_seq
