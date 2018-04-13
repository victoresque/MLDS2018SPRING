import torch
import torch.nn.functional as F
from torch.autograd import Variable

# TODO: check cross_entropy correctness


def cross_entropy(y_input, y_target):
    # batch size * max length * embedding size
    loss = Variable(torch.FloatTensor([0]), requires_grad=True)
    y_input = y_input.transpose(0, 1)
    y_target = y_target.transpose(0, 1)
    y_target = y_target.max(2)[1]
    seq_len = min(y_input.data.shape[0], y_target.data.shape[0])
    batch_size = y_input.data.shape[1]
    for i in range(seq_len):
        loss = loss + F.cross_entropy(y_input[i], y_target[i])
    # f.cross_entropy(y_input, y_target)
    loss = loss / batch_size
    return loss
