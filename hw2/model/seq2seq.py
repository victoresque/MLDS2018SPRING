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
    def __init__(self, input_size, hidden_size, output_size, embedder, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type.upper()
        self.embedder = embedder
        if self.rnn_type not in ['LSTM', 'GRU']:
            raise Exception('Unknown cell type: {}'.format(self.rnn_type))
        self.encoder = Encoder(self.input_size, self.hidden_size, rnn_type=self.rnn_type)
        self.decoder = Decoder(self.hidden_size, self.output_size, rnn_type=self.rnn_type)

    def forward(self, in_seq):
        enc_out, hidden = self.encoder(in_seq)
        out_seq = self.decoder(enc_out, hidden, self.embedder)
        return out_seq
