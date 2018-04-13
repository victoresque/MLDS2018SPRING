import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, mode='GRU', max_length=25, tokens=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mode = mode
        if mode.upper() == 'GRU':
            self.rnn = nn.GRU(output_size, hidden_size, batch_first=False,
                              num_layers=1, dropout=0, bidirectional=False)
        elif mode.upper() == 'LSTM':
            self.rnn = nn.LSTM(output_size, hidden_size, batch_first=False,
                               num_layers=1, dropout=0, bidirectional=False)
        else:
            raise Exception('Unknown cell type: {}'.format(mode))
        self.embedding = nn.Embedding(1, output_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.max_length = max_length
        self.tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3} \
            if tokens is None else tokens

    def forward(self, hidden):
        # output = self.embedding(in_seq).view(1, 1, -1)
        # output = F.relu(output)
        # output, hidden = self.rnn(output, hidden)
        out_seq = []

        def onehot(dim, label):
            v = np.zeros((dim,))
            v[label] = 1
            return v

        bos_onehot = onehot(self.output_size, self.tokens['<BOS>'])
        dec_in = Variable(torch.FloatTensor([[bos_onehot for _ in range(hidden.data.shape[1])]]))
        for i in range(self.max_length):
            if self.mode == 'GRU':
                dec_out, hidden = self.rnn(dec_in, hidden)
            else:
                dec_out, (hidden, _) = self.rnn(dec_in, hidden)
            dec_out = self.out(dec_out)
            # _, topi = dec_out.data.topk(1)
            # ni = topi[0].cpu().numpy()
            # out_seq.append(ni)
            out_seq.append(dec_out.cpu().data.numpy()[0])
            dec_in = dec_out
            # if ni == self.tokens['<EOS>']:
            #     break

        out_seq = torch.FloatTensor(np.array(out_seq))
        out_seq = Variable(out_seq.transpose(0, 1))
        return out_seq

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
