import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base.base_model import BaseModel
from model.modules import Encoder, Decoder

# TODO: Seq2Seq basic encoder/decoder
# TODO: Bidirectional
# TODO: Attention
# TODO: Schedule sampling
# TODO: (Optional) Stacked attention
# TODO: Teacher forcing
# TODO: Beam search
# DONE: Use GRU/LSTM or GRUCell/LSTMCell? --> Use GRU/LSTM


class Seq2Seq(BaseModel):
    def __init__(self, input_size=4096, hidden_size=512, output_size=1000, mode='GRU'):
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mode = mode
        self._build_model()

    def _build_model(self):
        self.encoder = Encoder(self.input_size, self.hidden_size, mode=self.mode)
        self.decoder = Decoder(self.hidden_size, self.output_size, mode=self.mode)
        # self.enc_h0 = Variable(self.encoder.init_hidden())
        # self.dec_h0 = Variable(self.decoder.init_hidden())

    def forward(self, in_seq):
        if self.mode == 'GRU':
            enc_out, hidden = self.encoder(in_seq)
        else:
            enc_out, hidden = self.encoder(in_seq)
        out_seq = self.decoder(hidden)
        return out_seq
