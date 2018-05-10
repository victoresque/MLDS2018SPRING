import torch
import torch.nn.functional as F


def cross_entropy(input, target, weight):
    """
    Average over sequences (each sequence contributes the same)

    input
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    target:
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    weight:
        type:  Variable
        shape: max sequence length in batch x batch size
    """
    loss = 0
    batch_size = target.data.shape[1]
    target = target.max(2)[1]
    input = input.transpose(0, 1)
    target = target.transpose(0, 1)
    weight = weight.transpose(0, 1)
    total_weight = torch.sum(weight, dim=1).cpu().data.numpy()
    for i in range(batch_size):
        partial_loss = F.cross_entropy(input[i], target[i], size_average=False, reduce=False)
        partial_loss = torch.mul(partial_loss, weight[i])
        loss = loss + torch.sum(partial_loss) / float(total_weight[i])
    return loss / batch_size


def cross_entropy2(input, target, weight):
    """
    Average over words (each word contributes the same)

    input
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    target:
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    weight:
        type:  Variable
        shape: max sequence length in batch x batch size
    """
    loss = 0
    batch_size = target.data.shape[1]
    target = target.max(2)[1]
    input = input.transpose(0, 1)
    target = target.transpose(0, 1)
    weight = weight.transpose(0, 1)
    word_count = torch.sum(torch.sum(weight, dim=1), dim=0).cpu().data.numpy().squeeze()
    for i in range(batch_size):
        partial_loss = F.cross_entropy(input[i], target[i], size_average=False, reduce=False)
        partial_loss = torch.mul(partial_loss, weight[i])
        loss = loss + torch.sum(partial_loss)
    return loss / float(word_count)


def cross_entropy3(input, target, weight):
    """
    Only average over batch size, does not consider sequence length

    input
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    target:
        type:  Variable
        shape: max sequence length in batch x batch size x emb size
    weight:
        type:  Variable
        shape: max sequence length in batch x batch size
    """
    loss = 0
    batch_size = target.data.shape[1]
    target = target.max(2)[1]
    input = input.transpose(0, 1)
    target = target.transpose(0, 1)
    weight = weight.transpose(0, 1)
    for i in range(batch_size):
        partial_loss = F.cross_entropy(input[i], target[i], size_average=False, reduce=False)
        partial_loss = torch.mul(partial_loss, weight[i])
        loss = loss + torch.sum(partial_loss)
    return loss / batch_size


def mse_loss(input, target, weight):
    """
        Average over sequences (each sequence contributes the same)

        input
            type:  Variable
            shape: max sequence length in batch x batch size x emb size
        target:
            type:  Variable
            shape: max sequence length in batch x batch size x emb size
        weight:
            type:  Variable
            shape: max sequence length in batch x batch size
        """
    loss = 0
    batch_size = target.data.shape[1]
    input = input.transpose(0, 1)
    target = target.transpose(0, 1)
    weight = weight.transpose(0, 1)
    total_weight = torch.sum(weight, dim=1).cpu().data.numpy()
    for i in range(batch_size):
        partial_loss = 0
        for j in range(int(total_weight[i])):
            partial_loss = partial_loss + F.mse_loss(input[i, j], target[i, j])
        loss = loss + partial_loss
    return loss / float(batch_size)
