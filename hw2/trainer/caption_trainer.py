import numpy as np
import torch
from torch.autograd import Variable
from base.base_trainer import BaseTrainer


class CaptionTrainer(BaseTrainer):
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, with_cuda, verbosity, identifier='',
                 valid_data_loader=None, logger=None):
        super(CaptionTrainer, self).__init__(model, loss, metrics, optimizer, epochs,
                                             save_dir, save_freq, resume, verbosity, identifier, logger)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader else False
        self.with_cuda = with_cuda

    def _train_epoch(self, epoch):
        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (in_seq, targ_seq, fmt) in enumerate(self.data_loader):
            in_seq, targ_seq = torch.FloatTensor(in_seq), torch.LongTensor(targ_seq)
            in_seq, targ_seq = Variable(in_seq), Variable(targ_seq)
            if self.with_cuda:
                in_seq, targ_seq = in_seq.cuda(), targ_seq.cuda()

            self.optimizer.zero_grad()
            out_seq = self.model(in_seq)
            loss = self.loss(out_seq, targ_seq)
            loss.backward()
            self.optimizer.step()

            for i, metric in enumerate(self.metrics):
                out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
                out_seq = np.transpose(out_seq, (1, 0, 2))
                out_seq = self.data_loader.embedder.decode_lines(out_seq)
                out_seq = dict((fmt[j]['id'], line) for j, line in enumerate(out_seq))
                total_metrics[i] += metric(out_seq, fmt)

            if batch_idx == 0:
                for k, v in out_seq.items():
                    print('{:30s}'.format(k), v)
                print('')

            total_loss += loss.data[0]
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(in_seq), len(self.data_loader) * len(in_seq),
                    100.0 * batch_idx / len(self.data_loader), loss.data[0]))

        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        log = {'loss': avg_loss, 'metrics': avg_metrics}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        for batch_idx, (in_seq, targ_seq, fmt) in enumerate(self.valid_data_loader):
            in_seq, targ_seq = torch.FloatTensor(in_seq), torch.LongTensor(targ_seq)

            in_seq, targ_seq = Variable(in_seq), Variable(targ_seq)
            if self.with_cuda:
                in_seq, targ_seq = in_seq.cuda(), targ_seq.cuda()

            out_seq = self.model(in_seq)
            loss = self.loss(out_seq, targ_seq)
            total_val_loss += loss.data[0]

            for i, metric in enumerate(self.metrics):
                out_seq = np.array([seq.data.cpu().numpy() for seq in out_seq])
                out_seq = np.transpose(out_seq, (1, 0, 2))
                out_seq = self.data_loader.embedder.decode_lines(out_seq)
                out_seq = dict((fmt[j]['id'], line) for j, line in enumerate(out_seq))
                total_val_metrics[i] += metric(out_seq, fmt)

            if batch_idx == 0:
                for k, v in out_seq.items():
                    print('{:30s}'.format(k), v)
                print('')

        avg_val_loss = total_val_loss / len(self.valid_data_loader)
        avg_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        return {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}
