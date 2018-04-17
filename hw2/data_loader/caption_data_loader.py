import os
import json
import random
import numpy as np
from base import BaseDataLoader

# FIXME: (important) Train only by the first label?
# DONE: Output sequence padding
# DONE: One-hot encoding
# DONE: Tokens (<PAD>, <BOS>, <EOS>, <UNK>, ...)


class CaptionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, embedder, emb_size, shuffle=True, mode='train'):
        shuffle = shuffle if mode == 'train' else False
        super(CaptionDataLoader, self).__init__(batch_size, shuffle)
        self.mode = mode
        self.in_seq = []
        self.out_seq = []
        self.formatted = []
        self.__parse_dataset(os.path.join(data_dir, 'MLDS_hw2_1_data'))

        self.embedder = embedder(self.corpus, emb_size=emb_size)
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
            self.out_seq.append(labels[video_id])
            self.formatted.append({'caption': labels[video_id], 'id': video_id})

        for video_id, captions in corpus_labels.items():
            self.corpus.extend(captions)

    def __next__(self):
        """
        Next batch
        :return:
            in_seq_batch:  80 x batch size x 4096
            out_seq_batch: sequence length x batch size x 1000
            formatted:     same format as in sample output
                           [{'caption': ['...', '...'], 'id': '...'}, ...]
        """
        batch = super(CaptionDataLoader, self).__next__()
        in_seq_batch, out_seq_batch, formatted_batch = batch
        # out_seq_batch = [random.choice(seq) for seq in out_seq_batch]
        out_seq_batch = [seq[0] for seq in out_seq_batch]
        out_seq_batch = pad_batch(out_seq_batch,
                                  self.embedder.encode_word('<PAD>'),
                                  self.embedder.encode_word('<EOS>'))
        in_seq_batch = np.array(in_seq_batch).transpose((1, 0, 2))
        out_seq_batch = np.array(out_seq_batch).transpose((1, 0, 2))
        if self.mode == 'train':
            return in_seq_batch, out_seq_batch, formatted_batch
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


def pad_batch(batch, pad_val, eos_val, seq_len=24):
    # FIXME: seq_len
    eos_val = eos_val.reshape((1, -1))
    for i, seq in enumerate(batch):
        seq = seq[:seq_len-1]
        seq = np.append(seq, eos_val, axis=0)
        if len(seq) < seq_len:
            batch[i] = np.append(seq, [pad_val for _ in range(seq_len-len(seq))], axis=0)
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

    data_loader = CaptionDataLoader('../datasets', 128, 60)
    for isb, osb, fb in data_loader:
        print(isb.shape)
        print(osb.shape)
        # print(fb)
        input()
