import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from tensorboardX import SummaryWriter
from base import BaseTrainer
from utils import *


def wasserstein_loss(pred, target):
    return torch.mean(pred * target)


def random_interpolate(real, fake):
    batch_size = real.size()[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(-1, real.nelement() // batch_size).contiguous().view(*real.size())
    alpha = to_variable(real.is_cuda, alpha)
    interpolated = alpha * real + (1 - alpha) * fake
    return interpolated


def gradient_penalty(pred, interpolated_images):
    gradients = autograd.grad(pred, interpolated_images,
                              grad_outputs=torch.ones(pred.size()).cuda() if pred.is_cuda else torch.ones(pred.size()),
                              create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class WGANGPTrainer(BaseTrainer):
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(WGANGPTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False

        self.gen_optimizer = self.optimizers['generator']
        self.dis_optimizer = self.optimizers['discriminator']
        self.noise_dim = model.generator.noise_dim

        # training tips
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
        for batch_idx, (real_images, ) in enumerate(self.data_loader):
            input_noise = torch.randn(self.batch_size, self.noise_dim, 1, 1)
            real_images = np.transpose(real_images, (0, 3, 1, 2))  # (batch, channel(BGR), width, height)

            real_target = -torch.ones(self.batch_size)
            fake_target = torch.ones(self.batch_size)

            # add noise to real images (tip 13)
            image_noise = torch.randn(*real_images.shape) * self.image_noise_var
            image_noise = self.image_noise_decay(image_noise, epoch)
            real_images = real_images + image_noise

            input_noise, real_target, fake_target, real_images = \
                to_variable(self.with_cuda, input_noise, real_target, fake_target, real_images)

            # training on discriminator
            fake_images = self.model.generator(input_noise)
            interpolated_images = random_interpolate(real_images, fake_images)
            interpolated_images = to_variable(self.with_cuda, interpolated_images)

            real_critic = self.model.discriminator(real_images)
            fake_critic = self.model.discriminator(fake_images)
            gp_critic = self.model.discriminator(interpolated_images)
            real_loss = wasserstein_loss(real_critic, real_target)
            fake_loss = wasserstein_loss(fake_critic, fake_target)
            gp_loss = gradient_penalty(gp_critic, interpolated_images)

            self.dis_optimizer.zero_grad()
            loss_d = real_loss + fake_loss + gp_loss * self.config['loss']['lambda']
            loss_d.backward()
            self.dis_optimizer.step()

            sum_loss_d += loss_d.data[0]
            n_loss_d += 1

            # fake images visualization
            show_grid(fake_images)

            # training on generator
            loss_g = None
            if batch_idx % self.dis_iter == 0:
                for i in range(self.gen_iter):
                    input_noise = torch.randn(*input_noise.size())
                    fake_target = -torch.ones(self.batch_size)
                    input_noise, fake_target = to_variable(self.with_cuda, input_noise, fake_target)
                    fake_images = self.model.generator(input_noise)
                    fake_critic = self.model.discriminator(fake_images)

                    self.gen_optimizer.zero_grad()
                    loss_g = wasserstein_loss(fake_critic, fake_target)
                    loss_g.backward()
                    self.gen_optimizer.step()

                    sum_loss_g += loss_g.data[0]
                    n_loss_g += 1

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
        for batch_idx, (real_images, ) in enumerate(self.data_loader):
            input_noise = torch.randn(self.batch_size, self.noise_dim, 1, 1)
            real_images = np.transpose(real_images, (0, 3, 1, 2))  # (batch, channel(BGR), width, height)

            real_target = -torch.ones(self.batch_size)
            fake_target = torch.ones(self.batch_size)

            # add noise to real images (tip 13)
            image_noise = torch.randn(*real_images.shape) * self.image_noise_var
            image_noise = self.image_noise_decay(image_noise, epoch)
            real_images = real_images + image_noise

            input_noise, real_target, fake_target, real_images = \
                to_variable(self.with_cuda, input_noise, real_target, fake_target, real_images)

            # training on discriminator
            fake_images = self.model.generator(input_noise)
            interpolated_images = random_interpolate(real_images, fake_images)
            interpolated_images = to_variable(self.with_cuda, interpolated_images)

            real_critic = self.model.discriminator(real_images)
            fake_critic = self.model.discriminator(fake_images)
            gp_critic = self.model.discriminator(interpolated_images)
            real_loss = wasserstein_loss(real_critic, real_target)
            fake_loss = wasserstein_loss(fake_critic, fake_target)
            gp_loss = gradient_penalty(gp_critic, interpolated_images)

            loss_d = real_loss + fake_loss + gp_loss * self.config['loss']['lambda']
            sum_loss_d += loss_d.data[0]
            n_loss_d += 1

            # generator
            loss_g = None
            if batch_idx % self.dis_iter == 0:
                for i in range(self.gen_iter):
                    input_noise = torch.randn(*input_noise.size())
                    fake_target = -torch.ones(self.batch_size)
                    input_noise, fake_target = to_variable(self.with_cuda, input_noise, fake_target)
                    fake_images = self.model.generator(input_noise)
                    fake_critic = self.model.discriminator(fake_images)

                    loss_g = wasserstein_loss(fake_critic, fake_target)
                    sum_loss_g += loss_g.data[0]
                    n_loss_g += 1

            full_loss.append({
                'iter': batch_idx,
                'loss_g': loss_g.data[0] if loss_g is not None else None,
                'loss_d': loss_d.data[0]
            })

            if self.verbosity >= 2:
                print_status(epoch, batch_idx, batch_idx+1,
                             len(self.data_loader), loss_d.data[0],
                             loss_g.data[0] if loss_g is not None else 0, mode='valid')

        log = {
            'val_loss': (sum_loss_g + sum_loss_d) / (n_loss_g + n_loss_d),
            'val_loss_g': sum_loss_g / n_loss_g,
            'val_loss_d': sum_loss_d / n_loss_d,
            'val_metrics': (total_metrics / n_loss_d).tolist(),
            'full_val_loss': full_loss
        }

        return log
