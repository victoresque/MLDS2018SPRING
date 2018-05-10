import numpy as np
import torch
from torch.autograd import Variable
from base.base_trainer import BaseTrainer
import sys


class CaptionTrainer(BaseTrainer):
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(CaptionTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))

    def _to_variable(self, in_seq, out_seq, targ_weight):
        in_seq = Variable(torch.FloatTensor(in_seq))
        out_seq = Variable(torch.FloatTensor(out_seq))
        targ_weight = Variable(torch.FloatTensor(targ_weight))
        if self.with_cuda:
            in_seq, out_seq, targ_weight = in_seq.cuda(), out_seq.cuda(), targ_weight.cuda()
        return in_seq, out_seq, targ_weight

    def _eval_metrics(self, out_seq, fmt):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
            out_seq = np.transpose(out_seq, (1, 0, 2))
            out_seq = self.data_loader.embedder.decode_lines(out_seq)
            out_seq = dict((fmt[j]['id'], line) for j, line in enumerate(out_seq))
            acc_metrics[i] += metric(out_seq, fmt)
        return acc_metrics

    def _show_seq(self, seq, fmt, status):
        seq = np.array([s.data.cpu().numpy() for s in seq])
        seq = np.transpose(seq, (1, 0, 2))
        seq = self.data_loader.embedder.decode_lines(seq)
        seq = dict((fmt[j]['id'], line) for j, line in enumerate(seq))
        print('')
        print(status)
        for i, (k, v) in enumerate(seq.items()):
            if i == 4:
                break
            print('{:30s}'.format(k), v)
        print('')

    def _train_epoch(self, epoch):
        self.model.train()
        if self.with_cuda:
            self.model.cuda()
        
        print("========================================================================")

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (in_seq, target) in enumerate(self.data_loader):
            targ_seq, targ_idx, targ_weight, formatted = target
            in_seq, targ_seq, targ_weight = self._to_variable(in_seq, targ_seq, targ_weight)

            self.optimizer.zero_grad()
            out_seq = self.model(in_seq, len(targ_seq), targ_seq, targ_idx, epoch)
            loss = self.loss(out_seq, targ_seq, targ_weight)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            total_metrics += self._eval_metrics(out_seq, formatted)

            if batch_idx == 0:
                self._show_seq(out_seq, formatted, 'train')
                self._show_seq(targ_seq, formatted, 'train_ground_truth')

            if self.verbosity >= 2:
                if batch_idx == 0: log_length = 0
                log_length = self.__print_status(epoch, batch_idx, batch_idx * in_seq.size(1),
                                                 len(self.data_loader) * in_seq.size(1),
                                                 loss.data[0], log_length)
                
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}
        

        return log

    def _valid_epoch(self):
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        for batch_idx, (in_seq, target) in enumerate(self.valid_data_loader):
            targ_seq, targ_idx, targ_weight, formatted = target
            in_seq, targ_seq, targ_weight = self._to_variable(in_seq, targ_seq, targ_weight)

            out_seq = self.model(in_seq, len(targ_seq))
            loss = self.loss(out_seq, targ_seq, targ_weight)
            total_val_loss += loss.data[0]
            total_val_metrics += self._eval_metrics(out_seq, formatted)

            if batch_idx == 0:
                self._show_seq(out_seq, formatted, 'valid')
                self._show_seq(targ_seq, formatted, 'valid_ground_truth')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
    
    def __print_status(self, epoch, batch_idx, n_trained, n_data, loss, pre_log_length):
        if batch_idx != 0:
            sys.stdout.write("\b"*pre_log_length) 
            sys.stdout.flush()
            
        log_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            epoch, n_trained, n_data, 100.0 * n_trained / n_data, loss)
        sys.stdout.write(log_msg)
        sys.stdout.flush()
        
        return len(log_msg)

