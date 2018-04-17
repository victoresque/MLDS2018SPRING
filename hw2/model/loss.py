import torch
import torch.nn.functional as F


def cross_entropy(input, target, mask):
    """
    input
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    target:
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    mask:
        type:  Variable
        shape: max sequence length in batch x batch size
    """
    loss = 0
    batch_size = target.data.shape[1]
    target = target.max(2)[1]
    input = input.transpose(0, 1)
    target = target.transpose(0, 1)
    mask = mask.transpose(0, 1)
    seq_len = torch.sum(mask, dim=1).cpu().data.numpy()
    for i in range(batch_size):
        loss = loss + F.cross_entropy(input[i], target[i], size_average=False) / float(seq_len[i])
    return loss / batch_size


def mse_loss(input, target, mask):
    loss = 0
    seq_len = target.data.shape[0]
    batch_size = target.data.shape[1]
    for i in range(seq_len):
        loss = loss + F.mse_loss(input[i], target[i])
    return loss / batch_size
