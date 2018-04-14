import torch
import torch.nn.functional as F
from torch.autograd import Variable

# TODO: (important) check cross_entropy correctness


def cross_entropy(y_input, y_target):
    # y_input:  max length * batch size * embedding size
    # y_target: max length * batch size
    loss = Variable(torch.FloatTensor([0]), requires_grad=True)
    is_cuda = y_target.is_cuda
    if is_cuda:
        loss = loss.cuda()
    emb_size = y_target.data.shape[2]
    y_target = y_target.transpose(0, 1)
    y_target = y_target.max(2)[1]
    batch_size = y_target.data.shape[1]

    y_input = torch.cat(y_input, 0)
    y_input = y_input.view(-1, batch_size, emb_size)
    seq_len = min(len(y_input), y_target.data.shape[0])
    for i in range(batch_size):
        loss = loss + F.cross_entropy(y_input[:seq_len, i, :], y_target[:seq_len, i], ignore_index=0)
    return loss / batch_size
