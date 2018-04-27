import os
import json
import random
import pickle
import numpy as np
from base import BaseDataLoader
from utils.util import ensure_dir


class ChatbotDataLoader(BaseDataLoader):
    def __init__(self, config, embedder, mode, path, embedder_path, vocab_path):
        assert mode == 'train' or mode == 'test'
        if mode == 'train':
            shuffle = config['data_loader']['shuffle']
            batch_size = config['data_loader']['batch_size']
        else:
            shuffle = False
            batch_size = 1
        super(ChatbotDataLoader, self).__init__(config, shuffle, batch_size)
        self.mode = mode
        self.in_seq = []
        self.out_seq = []
        self.__parse_dataset(path, embedder, embedder_path, vocab_path)

        if mode == 'train':
            ensure_dir(os.path.dirname(embedder_path))
            pickle.dump(self.embedder, open(embedder_path, 'wb'))

    def __parse_dataset(self, path, embedder, embedder_path, vocab_path):
        if self.mode == 'train':
            self.corpus = self.__load_corpus(os.path.join(path, 'clr_conversation.txt'))
            self.embedder = embedder(self.corpus, vocab_path, self.config)
            for i in range(len(self.corpus)-1):
                if self.corpus[i] != '+++$+++' and self.corpus[i+1] != '+++$+++':
                    self.in_seq.append(self.corpus[i])
                    self.out_seq.append(self.corpus[i+1])

            pack = list(zip(self.in_seq, self.out_seq))
            random.shuffle(pack)
            self.in_seq, self.out_seq = zip(*pack)
            seq_cnt = self.config['data_loader']['train_seq_count']
            self.in_seq = self.in_seq[:seq_cnt]
            self.out_seq = self.out_seq[:seq_cnt]
            ensure_dir(os.path.dirname(embedder_path))
            pickle.dump(self.embedder, open(embedder_path, 'wb'))
        else:
            pass

    def __next__(self):
        """
        Next batch

        in_seq_batch:
            type:  ndarray
            shape: max sequence length in input batch x batch size x emb size
        out_seq_batch:
            type:  ndarray
            shape: max sequence length in output batch x batch size x emb size
        out_seq_weight:
            type:  ndarray
            shape: max sequence length in batch x batch size
            note:  0 if <PAD>, else 1

        Note:
            should only return two items, output-related items packed into a tuple:
                input, (output, ...)
        """
        batch = super(ChatbotDataLoader, self).__next__()
        in_seq_batch, out_seq_batch = batch

        in_seq_batch = self.embedder.encode_lines(in_seq_batch)
        out_seq_batch = self.embedder.encode_lines(out_seq_batch)

        in_seq_batch = self.__pad_in(in_seq_batch, self.embedder.encode_word('<PAD>'))
        out_seq_batch, out_seq_weight = self.__pad_out(out_seq_batch,
                                                       self.embedder.encode_word('<PAD>'),
                                                       self.embedder.encode_word('<EOS>'))

        in_seq_batch = np.array(in_seq_batch).transpose((1, 0, 2))
        out_seq_batch = np.array(out_seq_batch).transpose((1, 0, 2))
        out_seq_weight = np.array(out_seq_weight).transpose((1, 0))
        if self.mode == 'train':
            return in_seq_batch, (out_seq_batch, out_seq_weight)
        else:
            return in_seq_batch

    def _pack_data(self):
        packed = list(zip(self.in_seq, self.out_seq))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.in_seq, self.out_seq = unpacked

    def _n_samples(self):
        return len(self.in_seq)

    def __load_corpus(self, path):
        """
        Load conversation corpus

        corpus:
            list of sentences
            each sentence is a str of chinese characters
        """
        with open(path, encoding='utf-8') as f:
            corpus = f.readlines()
        for i, line in enumerate(corpus):
            line = line.rstrip(' \n')
            line = ''.join(line.split())
            line = [word for word in line]
            corpus[i] = line
        return corpus

    def __pad_in(self, batch, pad_val):
        seq_len = 0
        for seq in batch:
            seq_len = max(len(seq), seq_len)
        for i, seq in enumerate(batch):
            seq = seq[:seq_len - 1]
            if len(seq) < seq_len:
                batch[i] = np.append([pad_val for _ in range(seq_len - len(seq))], seq, axis=0)
            else:
                batch[i] = seq

        return batch

    def __pad_out(self, batch, pad_val, eos_val):
        seq_len = 0
        for seq in batch:
            seq_len = max(len(seq), seq_len)
        weight = np.zeros((len(batch), seq_len))
        for i, seq in enumerate(batch):
            seq = seq[:seq_len-1]
            seq = np.append(seq, [eos_val], axis=0)
            if len(seq) < seq_len:
                batch[i] = np.append(seq, [pad_val for _ in range(seq_len-len(seq))], axis=0)
            else:
                batch[i] = seq

            for j, word in enumerate(seq):
                weight[i][j] = 1

        return batch, weight


if __name__ == '__main__':
    from preprocess.embedding import *
    config = dict()
    config['data_loader'] = dict()
    config['data_loader']['shuffle'] = True
    config['data_loader']['batch_size'] = 4
    config['data_loader']['weight_policy'] = 'bool'

    data_loader = \
        ChatbotDataLoader(config, ChineseOneHotEmbedder,
                          'train', '../datasets/MLDS_hw2_2_data/training_data', 'test_embedder.pkl')

    for batch in data_loader:
        in_seq, (out_seq, out_weight) = batch
        print(in_seq.shape)
        print(in_seq)
        print(out_seq.shape)
        print(out_seq)
        print(out_weight.shape)
        print(out_weight)
        input()
