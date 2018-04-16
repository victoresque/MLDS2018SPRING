import os
import json
import random
import numpy as np
from base import BaseDataLoader
from preprocess.embedding import OneHot

# FIXME: (important) Train only by the first label?
# DONE: Output sequence padding
# DONE: One-hot encoding
# DONE: Tokens (<PAD>, <BOS>, <EOS>, <UNK>, ...)


class CaptionDataLoader(BaseDataLoader):
    """ Format:
        :returns: in_seq, out_seq, formatted

        in_seq:     batch size * 80 * 4096
        out_seq:    batch size * max length * embedder dict size
            padded per batch in __next__()
        formatted:
            [{'caption': ['...', '...'], 'id': '...'}, ...]
    """
    def __init__(self, data_dir, batch_size, emb_size, shuffle=True, mode='train'):
        shuffle = shuffle if mode == 'train' else False
        super(CaptionDataLoader, self).__init__(batch_size, shuffle)
        self.mode = mode
        self.in_seq = []
        self.out_seq = []
        self.formatted = []
        self.__parse_dataset(os.path.join(data_dir, 'MLDS_hw2_1_data'))
        self.n_batch = len(self.in_seq) // self.batch_size
        self.batch_idx = 0
        self.embedder = OneHot(self.corpus, dict_size=emb_size)
        self.out_seq = [self.embedder.encode_lines(seq) for seq in self.out_seq]

    def __parse_dataset(self, base):
        self.video_ids = []
        self.corpus = []
        if self.mode == 'train':
            features = load_features(os.path.join(base, 'training_data/feat'))
            labels = load_labels(os.path.join(base, 'training_label.json'))
            corpus_labels = labels
        else:
            features = load_features(os.path.join(base, 'testing_data/feat'))
            labels = load_labels(os.path.join(base, 'testing_label.json'))
            corpus_labels = load_labels(os.path.join(base, 'training_label.json'))
        for video_id, feature in features.items():
            self.video_ids.append(video_id)
            self.in_seq.append(feature)
            # self.out_seq.append(labels[video_id][0])
            self.out_seq.append(labels[video_id])
            self.formatted.append({'caption': labels[video_id], 'id': video_id})

        for video_id, captions in corpus_labels.items():
            self.corpus.extend(captions)

    def __next__(self):
        batch = super(CaptionDataLoader, self).__next__()
        in_seq_batch, out_seq_batch, formatted_batch = batch
        out_seq_batch = [random.choice(seq) for seq in out_seq_batch]
        out_seq_batch = pad_batch(out_seq_batch,
                                  self.embedder.encode_word('<PAD>'),
                                  self.embedder.encode_word('<EOS>'))
        if self.mode == 'train':
            return np.array(in_seq_batch), np.array(out_seq_batch), formatted_batch
        else:
            return np.array(in_seq_batch), formatted_batch

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

    def __len__(self):
        self.n_batch = len(self.in_seq) // self.batch_size
        return self.n_batch


def load_features(path):
    features = {}
    filenames = os.listdir(path)
    for file in filenames:
        video_id, _ = os.path.splitext(file)
        feature = np.load(os.path.join(path, file))
        features[video_id] = feature
    return features


def load_labels(path):
    labels = {}
    raw_labels = json.load(open(path))
    for entry in raw_labels:
        labels[entry['id']] = entry['caption']
    return labels


def pad_batch(batch, pad_val, eos_val):
    maxlen = 0
    eos_val = eos_val.reshape((1, -1))
    for item in batch:
        maxlen = max(maxlen, len(item))
    for i, seq in enumerate(batch):
        seq = np.append(seq, eos_val, axis=0)
        if maxlen+1 != len(seq):
            batch[i] = np.append(seq, [pad_val for _ in range(maxlen-len(seq)+1)], axis=0)
        else:
            batch[i] = seq
    return batch


if __name__ == '__main__':
    # labels = load_labels('../datasets/MLDS_hw2_1_data/training_label.json')
    # for k, v in labels.items():
    #     print(k, v, end='\n\n')

    # features = load_features('../datasets/MLDS_hw2_1_data/training_data/feat')
    # for k, v in features.items():
    #     print(k, v, end='\n\n')

    data_loader = CaptionDataLoader('../datasets', 128)
    for isb, osb, fb in data_loader:
        print(isb.shape)
        print(osb.shape)
        # print(fb)
        input()
