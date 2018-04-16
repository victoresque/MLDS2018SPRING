import torch.nn.functional as F

# TODO: (important) check cross_entropy correctness


def cross_entropy(y_input, y_target, pad_token=0):
    # y_input:  max length * batch size * embedding size
    # y_target: max length * batch size
    loss = 0
    batch_size = y_target.data.shape[1]
    y_target = y_target.max(2)[1]
    for i in range(min(len(y_input), y_target.data.shape[0])):
        loss = loss + F.cross_entropy(y_input[i], y_target[i],
                                      size_average=False,
                                      ignore_index=pad_token)
    return loss / batch_size
