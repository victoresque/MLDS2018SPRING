import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base import BaseModel
from model.modules import Encoder, Decoder

# TODO: Bidirectional
# TODO: Attention
# TODO: Scheduled sampling
# TODO: Teacher forcing
# TODO: Beam search
# TODO: (Optional) Stacked attention


class Seq2Seq(BaseModel):
    """
    Note:
        input:
            type:  Variable
            shape: 80 x batch size x 4096
        output:
            type:  Variable
            shape: max sequence length in batch x batch size x emb size
    """
    def __init__(self, config, embedder):
        super(Seq2Seq, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.emb_size = config['emb_size']
        self.embedder = embedder
        self.rnn_type = config['rnn_type'].upper()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, in_seq, seq_len):
        enc_out, hidden = self.encoder(in_seq)
        out_seq = self.decoder(enc_out, hidden, seq_len, self.embedder)
        return out_seq
