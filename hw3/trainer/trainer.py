import numpy as np
import torch, sys
from torch.autograd import Variable
from base import BaseTrainer


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
        self.training_ratio = config['trainer']['gen-dis_training_ratio']
        #self.log_step = int(np.sqrt(self.batch_size))

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
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model['gen'].train()
        self.model['dis'].train()
        if self.with_cuda:
            self.model['gen'].cuda()
            self.model['dis'].cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (noise, real_images, labels) in enumerate(self.data_loader):
            noise = np.reshape(noise, (*noise.shape, 1, 1))
            real_images = np.transpose(real_images, (0, 3, 1, 2))
            noise, real_images, labels = self._to_variable(noise, real_images, labels)
            
            # training on discrimator
            self.dis_optimizer.zero_grad()
            gen_images = self.model['gen'](noise)
            conca_images = torch.cat((gen_images, real_images), dim=0)
            shuffle_idx = torch.LongTensor(np.random.permutation(2*self.batch_size))
            if self.with_cuda: shuffle_idx = shuffle_idx.cuda()
            images = conca_images[shuffle_idx]
            target = labels[shuffle_idx]

            output = self.model['dis'](images)
            loss_d = self.loss(output, target)
            loss_d.backward()
            self.dis_optimizer.step()
            
            total_loss += loss_d.data[0]
            total_metrics += self._eval_metrics(output, target)
                
            # training on generator
            loss_g = Variable(torch.zeros(1))
            if batch_idx % 10 > 10/(self.training_ratio+1):
                self.gen_optimizer.zero_grad()
                noise = torch.randn(*noise.size())
                noise = Variable(noise).cuda() if self.with_cuda else Variable(noise)
                gen_images = self.model['gen'](noise)
                output = self.model['dis'](gen_images)
                
                target = labels[self.batch_size:]          
                loss_g = self.loss(output, target)
                loss_g.backward()
                self.gen_optimizer.step()                

                total_loss += loss_g.data[0]
                total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2:
                log_length = self.__print_status(epoch, batch_idx, batch_idx+1, len(self.data_loader),
                                                 loss_d.data[0]+loss_g.data[0])

        log = {
            'loss': total_loss / len(self.data_loader), 
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = self._to_variable(data, target)

            output = self.model(data)
            loss = self.loss(output, target)

            total_val_loss += loss.data[0]
            total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader), 
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def __print_status(self, epoch, batch_idx, n_trained, n_data, loss):
        if batch_idx == 0: print("")
        log_msg = '\rTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            epoch, n_trained, n_data, 100.0 * n_trained / n_data, loss)
        sys.stdout.write(log_msg)
        sys.stdout.flush()