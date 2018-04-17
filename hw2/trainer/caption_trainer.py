import numpy as np
import torch
from torch.autograd import Variable
from base.base_trainer import BaseTrainer


class CaptionTrainer(BaseTrainer):
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, with_cuda, verbosity, training_name='',
                 valid_data_loader=None, train_logger=None, monitor='loss', monitor_mode='min'):
        super(CaptionTrainer, self).__init__(model, loss, metrics, optimizer, epochs,
                                             save_dir, save_freq, resume, verbosity, training_name,
                                             with_cuda, train_logger, monitor, monitor_mode)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))

    def _to_variable(self, in_seq, out_seq):
        in_seq, out_seq = torch.FloatTensor(in_seq), torch.FloatTensor(out_seq)
        in_seq, out_seq = Variable(in_seq), Variable(out_seq)
        if self.with_cuda:
            in_seq, out_seq = in_seq.cuda(), out_seq.cuda()
        return in_seq, out_seq

    def _eval_metrics(self, out_seq, fmt):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
            out_seq = np.transpose(out_seq, (1, 0, 2))
            out_seq = self.data_loader.embedder.decode_lines(out_seq)
            out_seq = dict((fmt[j]['id'], line) for j, line in enumerate(out_seq))
            acc_metrics[i] += metric(out_seq, fmt)
        return acc_metrics

    def _show_seq(self, seq, fmt):
        seq = np.array([s.data.cpu().numpy() for s in seq])
        seq = np.transpose(seq, (1, 0, 2))
        seq = self.data_loader.embedder.decode_lines(seq)
        seq = dict((fmt[j]['id'], line) for j, line in enumerate(seq))
        for i, (k, v) in enumerate(seq.items()):
            if i == 4: break
            print('{:30s}'.format(k), v)
        print('--------------------------------------------')

    def _train_epoch(self, epoch):
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (in_seq, targ_seq, fmt) in enumerate(self.data_loader):
            in_seq, targ_seq = self._to_variable(in_seq, targ_seq)

            self.optimizer.zero_grad()
            out_seq = self.model(in_seq)
            loss = self.loss(out_seq, targ_seq)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            total_metrics += self._eval_metrics(out_seq, fmt)

            if batch_idx == 0:
                self._show_seq(out_seq, fmt)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(in_seq), len(self.data_loader) * len(in_seq),
                    100.0 * batch_idx / len(self.data_loader), loss.data[0]))

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
        for batch_idx, (in_seq, targ_seq, fmt) in enumerate(self.valid_data_loader):
            in_seq, targ_seq = self._to_variable(in_seq, targ_seq)

            out_seq = self.model(in_seq)
            loss = self.loss(out_seq, targ_seq)
            total_val_loss += loss.data[0]
            total_val_metrics += self._eval_metrics(out_seq, fmt)

            if batch_idx == 0:
                self._show_seq(out_seq, fmt)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
