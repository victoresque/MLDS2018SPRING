import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from base import BaseTrainer
from utils import *


# modified generator loss (tip 02)
def modified_generator_loss(pred, _):
    return -torch.mean(torch.log(pred))


class CGANTrainer(BaseTrainer):
    def __init__(self, model, loss, metrics, vis, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(CGANTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.vis = vis

        self.gen_optimizer = self.optimizers['generator']
        self.dis_optimizer = self.optimizers['discriminator']
        self.noise_dim = model.generator.noise_dim

        # training tips
        self.generator_loss = modified_generator_loss if config['tips']['02']['enabled'] else F.binary_cross_entropy
        self.image_noise_var, self.image_noise_decay = (config['tips']['13']['config']['var'],
                                                        eval(config['tips']['13']['config']['decay'])) \
            if config['tips']['13']['enabled'] else (0, eval('lambda x, epoch: x'))
        self.gen_iter, self.dis_iter = (config['tips']['14']['config']['gen_iter'],
                                        config['tips']['14']['config']['dis_iter']) \
            if config['tips']['14']['enabled'] else (1, 1)

        # tensorboard configuration
        self.writer = SummaryWriter('../result_tensorboard')

    def _train_epoch(self, epoch):
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        full_loss = []
        sum_loss_g, n_loss_g = 0, 0
        sum_loss_d, n_loss_d = 0, 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (real_images, real_conditions) in enumerate(self.data_loader):
            input_noise = torch.randn(self.batch_size, self.noise_dim, 1, 1)
            real_images = np.transpose(real_images, (0, 3, 1, 2))  # (batch, channel(BGR), width, height)

            real_labels = torch.ones(self.batch_size)
            fake_labels = torch.zeros(self.batch_size)

            # add noise to real images (tip 13)
            image_noise = torch.randn(*real_images.shape) * self.image_noise_var
            image_noise = self.image_noise_decay(image_noise, epoch)
            real_images = real_images + image_noise

            input_noise, real_images, real_labels, fake_labels, real_conditions = \
                to_variable(self.with_cuda, input_noise, real_images, real_labels, fake_labels, real_conditions)

            # training on discriminator
            fake_images = self.model.generator(input_noise, real_conditions)
            real_critic = self.model.discriminator(real_images, real_conditions)
            fake_critic = self.model.discriminator(fake_images, real_conditions)
            real_loss = F.binary_cross_entropy(real_critic, real_labels)
            fake_loss = F.binary_cross_entropy(fake_critic, fake_labels)

            self.dis_optimizer.zero_grad()
            loss_d = (real_loss + fake_loss) / 2
            loss_d.backward()
            self.dis_optimizer.step()

            sum_loss_d += loss_d.data[0]
            total_metrics += eval_metrics(self.metrics, real_critic, real_labels) / 2
            total_metrics += eval_metrics(self.metrics, fake_critic, fake_labels) / 2
            n_loss_d += 1

            # fake images visualization
            if self.vis:
                show_grid(fake_images)

            # training on generator
            loss_g = None
            if batch_idx % self.dis_iter == 0:
                for i in range(self.gen_iter):
                    input_noise = torch.randn(*input_noise.size())
                    fake_target = torch.ones(self.batch_size)
                    input_noise, fake_target = to_variable(self.with_cuda, input_noise, fake_target)
                    fake_images = self.model.generator(input_noise, real_conditions)
                    fake_critic = self.model.discriminator(fake_images, real_conditions)

                    self.gen_optimizer.zero_grad()
                    loss_g = self.generator_loss(fake_critic, fake_target)
                    loss_g.backward()
                    self.gen_optimizer.step()

                    sum_loss_g += loss_g.data[0]
                    n_loss_g += 1

                    # self.writer.add_image('image_result', grid, epoch)

            full_loss.append({
                'iter': batch_idx,
                'loss_g': loss_g.data[0] if loss_g is not None else None,
                'loss_d': loss_d.data[0]
            })

            if self.verbosity >= 2:
                print_status(epoch, batch_idx, batch_idx+1,
                             len(self.data_loader), loss_d.data[0],
                             loss_g.data[0] if loss_g is not None else 0)

        log = {
            'loss': (sum_loss_g + sum_loss_d) / (n_loss_g + n_loss_d),
            'loss_g': sum_loss_g / n_loss_g,
            'loss_d': sum_loss_d / n_loss_d,
            'metrics': (total_metrics / n_loss_d).tolist(),
            'full_loss': full_loss
        }

        if self.valid:
            val_log = self._valid_epoch(epoch)
            return {**log, **val_log}
        else:
            return log

    def _valid_epoch(self, epoch):
        self.model.eval()

        if self.with_cuda:
            self.model.cuda()

        full_loss = []
        sum_loss_g, n_loss_g = 0, 0
        sum_loss_d, n_loss_d = 0, 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (real_images, real_conditions) in enumerate(self.valid_data_loader):
            input_noise = torch.randn(self.batch_size, self.noise_dim, 1, 1)
            real_images = np.transpose(real_images, (0, 3, 1, 2))  # (batch, channel(BGR), width, height)

            real_labels = torch.ones(self.batch_size)
            fake_labels = torch.zeros(self.batch_size)

            # add noise to real images (tip 13)
            image_noise = torch.randn(*real_images.shape) * self.image_noise_var
            image_noise = self.image_noise_decay(image_noise, epoch)
            real_images = real_images + image_noise

            input_noise, real_images, real_labels, fake_labels, real_conditions = \
                to_variable(self.with_cuda, input_noise, real_images, real_labels, fake_labels, real_conditions)

            # training on discriminator
            fake_images = self.model.generator(input_noise, real_conditions)
            real_critic = self.model.discriminator(real_images, real_conditions)
            fake_critic = self.model.discriminator(fake_images, real_conditions)
            real_loss = F.binary_cross_entropy(real_critic, real_labels)
            fake_loss = F.binary_cross_entropy(fake_critic, fake_labels)

            loss_d = (real_loss + fake_loss) / 2
            sum_loss_d += loss_d.data[0]
            total_metrics += eval_metrics(self.metrics, real_critic, real_labels) / 2
            total_metrics += eval_metrics(self.metrics, fake_critic, fake_labels) / 2
            n_loss_d += 1

            # generator
            loss_g = None
            if batch_idx % self.dis_iter == 0:
                for i in range(self.gen_iter):
                    input_noise = torch.randn(*input_noise.size())
                    fake_target = torch.ones(self.batch_size)
                    input_noise, fake_target = to_variable(self.with_cuda, input_noise, fake_target)
                    fake_images = self.model.generator(input_noise, real_conditions)
                    fake_critic = self.model.discriminator(fake_images, real_conditions)

                    loss_g = self.generator_loss(fake_critic, fake_target)
                    sum_loss_g += loss_g.data[0]
                    n_loss_g += 1

            full_loss.append({
                'iter': batch_idx,
                'loss_g': loss_g.data[0] if loss_g is not None else None,
                'loss_d': loss_d.data[0]
            })

            if self.verbosity >= 2:
                print_status(epoch, batch_idx, batch_idx+1,
                             len(self.valid_data_loader), loss_d.data[0],
                             loss_g.data[0] if loss_g is not None else 0, mode='valid')

        log = {
            'val_loss': (sum_loss_g + sum_loss_d) / (n_loss_g + n_loss_d),
            'val_loss_g': sum_loss_g / n_loss_g,
            'val_loss_d': sum_loss_d / n_loss_d,
            'val_metrics': (total_metrics / n_loss_d).tolist(),
            'full_val_loss': full_loss
        }

        return log
