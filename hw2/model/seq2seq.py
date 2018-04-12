from base.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

# TODO: Use GRU/LSTM or GRUCell/LSTMCell?
# TODO: Seq2Seq base encoder/decoder
# TODO: Bidirectional
# TODO: Attention
# TODO: Schedule sampling
# TODO: Stacked attention
# TODO: Teacher forcing
# TODO: Beam search


class Seq2Seq(BaseModel):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.encoder = None
        self.decoder = None
        self.build_model()

    def build_model(self):
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return F.log_softmax(output, dim=1)
