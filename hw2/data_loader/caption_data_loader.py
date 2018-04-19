import os
import json
import random
import numpy as np
from base import BaseDataLoader


class CaptionDataLoader(BaseDataLoader):
    def __init__(self, config, embedder, mode):
        assert mode == 'train' or mode == 'test'
        if mode == 'train':
            config['data_loader']['shuffle'] = True
        else:
            config['data_loader']['batch_size'] = 1
        super(CaptionDataLoader, self).__init__(config)
        self.mode = mode
        self.in_seq = []
        self.out_seq = []
        self.formatted = []
        self.sample_range = config['data_loader']['sample_range']
        self.__parse_dataset(os.path.join(config['data_loader']['data_dir'], 'MLDS_hw2_1_data'))

        self.embedder = embedder(self.corpus, config)
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

        in_seq_batch:
            type:  ndarray
            shape: 80 x batch size x 4096
        out_seq_batch:
            type:  ndarray
            shape: max sequence length in batch x batch size x emb size
        out_seq_mask:
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

        out_seq_batch, out_seq_mask = pad_batch(out_seq_batch,
                                                self.embedder.encode_word('<PAD>'),
                                                self.embedder.encode_word('<EOS>'))
        in_seq_batch = np.array(in_seq_batch).transpose((1, 0, 2))
        out_seq_batch = np.array(out_seq_batch).transpose((1, 0, 2))
        out_seq_mask = np.array(out_seq_mask).transpose((1, 0))
        if self.mode == 'train':
            return in_seq_batch, (out_seq_batch, out_seq_mask, formatted_batch)
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


def pad_batch(batch, pad_val, eos_val):
    seq_len = 0
    for seq in batch:
        seq_len = max(len(seq), seq_len)
    mask = np.ones((len(batch), seq_len))
    for i, seq in enumerate(batch):
        seq = seq[:seq_len-1]
        seq = np.append(seq, [eos_val], axis=0)
        if len(seq) < seq_len:
            batch[i] = np.append(seq, [pad_val for _ in range(seq_len-len(seq))], axis=0)
        else:
            batch[i] = seq
        for j in range(len(seq), seq_len):
            mask[i][j] = 0
    return batch, mask


if __name__ == '__main__':
    # labels = load_labels('../datasets/MLDS_hw2_1_data/training_label.json')
    # for k, v in labels.items():
    #     print(k, v, end='\n\n')

    # features = load_features('../datasets/MLDS_hw2_1_data/training_data/feat')
    # for k, v in features.items():
    #     print(k, v, end='\n\n')
    from preprocess.embedding import OneHotEmbedder
    data_loader = CaptionDataLoader('../datasets', 8, OneHotEmbedder, 1000)
    for isb, (osb, mask, fb) in data_loader:
        print(isb.shape)
        print(osb.shape)
        print(mask.shape)
        # print(fb)
        input()
