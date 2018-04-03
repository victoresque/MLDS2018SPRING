import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    checkpoint = torch.load('../../models/saved/1-3-1/DeeperMnistCNN64_mnist_checkpoint_epoch500_loss_0.16779.pth.tar')
    logger = checkpoint['logger']
    x = [entry['epoch'] for _, entry in logger.entries.items()]

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    y1 = [entry['loss'] for _, entry in logger.entries.items()]
    y2 = [entry['val_loss'] for _, entry in logger.entries.items()]
    plt.plot(x, y1, 'r', label='train')
    plt.plot(x, y2, 'b', label='test')
    plt.grid()
    plt.title('Training/testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')

    plt.subplot(122)
    y1 = [entry['accuracy'] for _, entry in logger.entries.items()]
    y2 = [entry['val_accuracy'] for _, entry in logger.entries.items()]
    plt.plot(x, y1, 'r', label='train')
    plt.plot(x, y2, 'b', label='test')
    plt.grid()
    plt.title('Training/testing accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
