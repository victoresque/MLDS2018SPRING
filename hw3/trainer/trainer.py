import sys
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from base import BaseTrainer
import cv2


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

        self.gen_optimizer = self.optimizers['generator']
        self.dis_optimizer = self.optimizers['discriminator']

        for key, value in config['tips'].items():
            if key.startswith('13'):
                self.image_noise_var, self.image_noise_decay = (value['config']['var'],
                                                                eval(value['config']['decay'])) \
                    if value['enabled'] else (0, eval('lambda x, epoch: x'))
            elif key.startswith('14'):
                self.gen_iter, self.dis_iter = (value['config']['gen_iter'],
                                                value['config']['dis_iter']) if value['enabled'] else (1, 1)

        # tensorboard configuration
        self.writer = SummaryWriter('../result_tensorboard')

    def _to_variable(self, *args):
        return_var = []
        for data in args:
            return_var.append(Variable(torch.FloatTensor(data)))
        if self.with_cuda:
            return_var = [data.cuda() for data in return_var]
        return return_var

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        output = np.array([1 if i > 0.5 else 0 for i in output])
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        self.model.generator.train()
        self.model.discriminator.train()
        if self.with_cuda:
            self.model.generator.cuda()
            self.model.discriminator.cuda()

        sum_loss_g, n_loss_g = 0, 0
        sum_loss_d, n_loss_d = 0, 0

        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (noise, real_images, labels) in enumerate(self.data_loader):
            noise = np.reshape(noise, (*noise.shape, 1, 1))
            real_images = np.transpose(real_images, (0, 3, 1, 2))  # (batch, channel(BGR), width, height)
            noise, real_images, labels = self._to_variable(noise, real_images, labels)

            # add noise to real images (tip 13)
            image_noise = torch.randn(*real_images.size()) * self.image_noise_var
            image_noise = self.image_noise_decay(image_noise, epoch)
            image_noise = Variable(image_noise)
            image_noise = image_noise.cuda() if self.with_cuda else image_noise
            real_images = real_images + image_noise

            # training on discriminator
            # real part
            self.dis_optimizer.zero_grad()
            real_output = self.model.discriminator(real_images)
            real_loss = self.loss(real_output, labels[self.batch_size:])
            total_metrics += self._eval_metrics(real_output, labels[self.batch_size:]) * 0.5

            # fake part
            gen_images = self.model.generator(noise)

            # training process visualization
            imgs = (gen_images.cpu().data[:64] + 1)/2
            grid = torchvision.utils.make_grid(imgs).numpy()
            grid = np.transpose(grid, (1, 2, 0))
            cv2.imshow('generated images', grid)
            cv2.waitKey(1)

            fake_output = self.model.discriminator(gen_images)
            fake_loss = self.loss(fake_output, labels[:self.batch_size])

            loss_d = real_loss + fake_loss  # / 2

            loss_d.backward()
            self.dis_optimizer.step()

            sum_loss_d += loss_d.data[0]
            n_loss_d += 1
            total_metrics += self._eval_metrics(fake_output, labels[:self.batch_size]) * 0.5

            # training on generator
            if batch_idx % self.dis_iter == 0:
                for i in range(self.gen_iter):
                    self.gen_optimizer.zero_grad()
                    noise = torch.randn(*noise.size())
                    noise = Variable(noise).cuda() if self.with_cuda else Variable(noise)
                    gen_images = self.model.generator(noise)
                    output = self.model.discriminator(gen_images)

                    target = labels[self.batch_size:]
                    loss_g = self.loss(output, target)
                    loss_g.backward()
                    self.gen_optimizer.step()

                    sum_loss_g += loss_g.data[0]
                    n_loss_g += 1
                    total_metrics += self._eval_metrics(output, target)

                    # self.writer.add_image('image_result', grid, epoch)

            if self.verbosity >= 2:
                self.__print_status(epoch, batch_idx, batch_idx+1, len(self.data_loader),
                                    loss_d.data[0], loss_g.data[0])

        log = {
            'loss': (sum_loss_g + sum_loss_d) / (n_loss_g + n_loss_d),
            'loss_g': sum_loss_g / n_loss_g,
            'loss_d': sum_loss_d / n_loss_d,
            'metrics': (total_metrics / (len(self.data_loader))).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        self.model.generator.eval()
        self.model.discriminator.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        result = torch.FloatTensor()
        for batch_idx, (noise, real_images, labels) in enumerate(self.valid_data_loader):
            noise = np.reshape(noise, (*noise.shape, 1, 1))
            real_images = np.transpose(real_images, (0, 3, 1, 2))
            noise, real_images, labels = self._to_variable(noise, real_images, labels)

            gen_images = self.model.generator(noise)
            images = torch.cat((gen_images, real_images), dim=0)
            output = self.model.discriminator(images)
            loss = self.loss(output, labels)

            result = torch.cat((result, gen_images.cpu().data), dim=0)

            total_val_loss += loss.data[0]
            total_val_metrics += self._eval_metrics(output, labels)

        # for tensorboard visualization
        # grid = torchvision.utils.make_grid(result)
        # self.writer.add_image('image_result', grid, epoch)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def __print_status(self, epoch, batch_idx, n_trained, n_data, loss_d, loss_g):
        if batch_idx == 0:
            print('')
        log_msg = '\rTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: D:{:.6f}, G:{:.6f}'.format(
            epoch, n_trained, n_data, 100.0 * n_trained / n_data, loss_d, loss_g)
        sys.stdout.write(log_msg)
        sys.stdout.flush()
        if batch_idx == n_data-1:
            print('')
