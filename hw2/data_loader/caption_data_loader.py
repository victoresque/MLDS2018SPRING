import os
import json
import random
import pickle
import numpy as np
from base import BaseDataLoader
from utils.util import ensure_dir


class CaptionDataLoader(BaseDataLoader):
    def __init__(self, config, embedder, mode, path, embedder_path):
        assert mode == 'train' or mode == 'test'
        if mode == 'train':
            shuffle = config['data_loader']['shuffle']
            batch_size = config['data_loader']['batch_size']
        else:
            shuffle = False
            batch_size = 1
        super(CaptionDataLoader, self).__init__(config, shuffle, batch_size)
        self.mode = mode
        self.in_seq = []
        self.out_seq = []
        self.formatted = []
        self.sample_range = config['data_loader']['sample_range']
        self.__parse_dataset(path, embedder, embedder_path)
        self.out_seq = [self.embedder.encode_lines(seq) for seq in self.out_seq]

    def __parse_dataset(self, path, embedder, embedder_path):
        self.video_ids = []
        self.corpus = []
        if self.mode == 'train':
            features = self.__load_features(os.path.join(path, 'feat'))
            labels = self.__load_labels(os.path.join(path, '../training_label.json'))
            corpus_labels = labels
            for _, captions in corpus_labels.items():
                self.corpus.extend(captions)
            self.embedder = embedder(self.corpus, self.config)
            ensure_dir(os.path.dirname(embedder_path))
            pickle.dump(self.embedder, open(embedder_path, 'wb'))
        else:
            features = self.__load_features(os.path.join(path, 'feat'))
            labels = self.__load_labels(os.path.join(path, '../testing_label.json'))
            self.embedder = pickle.load(open(embedder_path, 'rb'))

        for video_id, feature in features.items():
            self.video_ids.append(video_id)
            self.in_seq.append(feature)
            self.out_seq.append(labels[video_id])
            self.formatted.append({'caption': labels[video_id], 'id': video_id})

    def __next__(self):
        """
        Next batch

        in_seq_batch:
            type:  ndarray
            shape: 80 x batch size x 4096
        out_seq_batch:
            type:  ndarray
            shape: max sequence length in batch x batch size x emb size
        out_seq_weight:
            type:  ndarray
            shape: max sequence length in batch x batch size
            note:  0 if <PAD>, else 1
        formatted:
            type:  list
            note:  same format as in sample output
                   [{'caption': ['...', '...'], 'id': '...'}, ...]

        Note:
            should only return two items, output-related items packed into a tuple:
                input, (output, ...)
        """
        batch = super(CaptionDataLoader, self).__next__()
        in_seq_batch, out_seq_batch, formatted_batch = batch

        sample_count = np.random.randint(self.sample_range[0], self.sample_range[1]+1)
        for i, seq in enumerate(in_seq_batch):
            rand_idx = sorted(np.random.permutation(80)[:sample_count])
            in_seq_batch[i] = [seq[idx] for idx in rand_idx]

        # pick random sequence as target
        out_seq_batch = [random.choice(seq) for seq in out_seq_batch]
        # pick first sequence as target
        # out_seq_batch = [seq[0] for seq in out_seq_batch]

        out_seq_batch, out_seq_weight = self.__pad_batch(out_seq_batch,
                                                         self.embedder.encode_word('<PAD>'),
                                                         self.embedder.encode_word('<EOS>'))

        in_seq_batch = np.array(in_seq_batch).transpose((1, 0, 2))
        out_seq_batch = np.array(out_seq_batch).transpose((1, 0, 2))
        out_seq_weight = np.array(out_seq_weight).transpose((1, 0))
        if self.mode == 'train':
            return in_seq_batch, (out_seq_batch, out_seq_weight, formatted_batch)
        else:
            return in_seq_batch, formatted_batch

    def _pack_data(self):
        packed = list(zip(self.in_seq, self.out_seq, self.formatted))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.in_seq, self.out_seq, self.formatted = unpacked

    def _n_samples(self):
        return len(self.in_seq)

    def __load_features(self, path):
        features = {}
        filenames = os.listdir(path)
        for file in filenames:
            video_id, _ = os.path.splitext(file)
            feature = np.load(os.path.join(path, file))
            features[video_id] = feature
        return features

    def __load_labels(self, path):
        labels = {}
        raw_labels = json.load(open(path))
        for entry in raw_labels:
            captions = entry['caption']
            if self.config['data_loader']['unique']:
                unique_captions = []
                for i, caption in enumerate(captions):
                    caption = caption.lower().rstrip('. ')
                    if caption not in unique_captions:
                        unique_captions.append(caption)
                labels[entry['id']] = unique_captions
            else:
                labels[entry['id']] = entry['caption']
        return labels

    def __pad_batch(self, batch, pad_val, eos_val):
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

            weight_policy = self.config['data_loader']['weight_policy']
            assert weight_policy == 'bool' or weight_policy == 'log_margin'
            if weight_policy == 'bool':
                for j, word in enumerate(seq):
                    weight[i][j] = 1
            elif weight_policy == 'log_margin':
                for j, word in enumerate(seq):
                    word = self.embedder.decode_word(word)
                    if word == '<EOS>':
                        weight[i][j] = 1
                    elif word == '<UNK>':
                        weight[i][j] = 0
                    else:
                        weight[i][j] = 1 / (2 + np.log10(self.embedder.frequency.get(word, np.inf)))

        return batch, weight
