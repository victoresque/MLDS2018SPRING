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
# DONE: Use GRU/LSTM or GRUCell/LSTMCell? --> Use GRU/LSTM
# DONE: Seq2Seq basic encoder/decoder


class Seq2Seq(BaseModel):
    def __init__(self, input_size=4096, hidden_size=256, output_size=1000, rnn_type='LSTM'):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type.upper()
        if self.rnn_type not in ['LSTM', 'GRU']:
            raise Exception('Unknown cell type: {}'.format(self.rnn_type))
        self.encoder = Encoder(self.input_size, self.hidden_size, rnn_type=self.rnn_type)
        self.decoder = Decoder(self.hidden_size, self.output_size, rnn_type=self.rnn_type)

    def forward(self, in_seq):
        enc_out, hidden = self.encoder(in_seq)
        out_seq = self.decoder(hidden)
        return out_seq
