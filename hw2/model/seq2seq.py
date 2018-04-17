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
    def __init__(self, config, embedder):
        super(Seq2Seq, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.emb_size = config['emb_size']
        self.embedder = embedder
        self.rnn_type = config['rnn_type'].upper()
        if self.rnn_type not in ['LSTM', 'GRU']:
            raise Exception('Unknown cell type: {}'.format(self.rnn_type))
        self.encoder = Encoder(self.input_size, self.hidden_size, rnn_type=self.rnn_type)
        self.decoder = Decoder(self.hidden_size, self.emb_size, rnn_type=self.rnn_type)

    def forward(self, in_seq):
        enc_out, hidden = self.encoder(in_seq)
        out_seq = self.decoder(enc_out, hidden, self.embedder)
        return out_seq
