import torch
import torch.nn.functional as F


def cross_entropy(input, target, mask):
    """
    Average over sequences (each sequence contributes the same)

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
        partial_loss = F.cross_entropy(input[i], target[i], size_average=False, reduce=False)
        partial_loss = torch.mul(partial_loss, mask[i])
        loss = loss + torch.sum(partial_loss) / float(seq_len[i])
    return loss / batch_size


def cross_entropy2(input, target, mask):
    """
    Average over words (each word contributes the same)

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
    word_count = torch.sum(torch.sum(mask, dim=1), dim=0).cpu().data.numpy().squeeze()
    for i in range(batch_size):
        partial_loss = F.cross_entropy(input[i], target[i], size_average=False, reduce=False)
        partial_loss = torch.mul(partial_loss, mask[i])
        loss = loss + torch.sum(partial_loss)
    return loss / float(word_count)


def mse_loss(input, target, mask):
    loss = 0
    seq_len = target.data.shape[0]
    batch_size = target.data.shape[1]
    for i in range(seq_len):
        loss = loss + F.mse_loss(input[i], target[i])
    return loss / batch_size
