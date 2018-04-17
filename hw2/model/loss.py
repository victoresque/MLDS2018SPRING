import torch.nn.functional as F


def cross_entropy(y_input, y_target):
    # y_input:  max length * batch size * onehot embedding size
    # y_target: max length * batch size * onehot embedding size
    loss = 0
    seq_len = y_target.data.shape[0]
    batch_size = y_target.data.shape[1]
    y_target = y_target.max(2)[1]
    for i in range(seq_len):
        loss = loss + F.cross_entropy(y_input[i], y_target[i])
    return loss / batch_size


def mse_loss(y_input, y_target):
    # y_input:  max length * batch size * w2v embedding size
    # y_target: max length * batch size * w2v embedding size
    loss = 0
    seq_len = y_target.data.shape[0]
    batch_size = y_target.data.shape[1]
    for i in range(seq_len):
        loss = loss + F.mse_loss(y_input[i], y_target[i])
    return loss / batch_size
