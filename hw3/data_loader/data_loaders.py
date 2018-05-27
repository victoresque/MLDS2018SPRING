import torch, cv2, os, pickle
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import Embedder, ensure_dir


class GanDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(GanDataLoader, self).__init__(config)
        self.data_dir1 = config['data_loader']['data_dir1']
        self.data_dir2 = config['data_loader']['data_dir2']
        self.images = []
        self.__parse_images()

    def __parse_images(self):
        # add images in data_dir1
        for img_file in sorted(os.listdir(self.data_dir1), key=lambda x: int(x[:x.rfind('.')])):
            img_path = os.path.join(self.data_dir1, img_file)
            img = cv2.imread(img_path)
            if img.shape != (64, 64, 3):
                img = cv2.resize(img, (64, 64))
            self.images.append(img)
        # add images in data_dir2
        for img_file in sorted(os.listdir(self.data_dir2), key=lambda x: int(x[:x.rfind('.')])):
            img_path = os.path.join(self.data_dir2, img_file)
            img = cv2.imread(img_path)
            if img.shape != (64, 64, 3):
                img = cv2.resize(img, (64, 64))
            self.images.append(img)
        self.images = (np.array(self.images) - 127.5)/127.5

    def __iter__(self):
        super(GanDataLoader, self).__iter__()
        return self

    def __next__(self):
        """
        :return: img
            shape = (batch_size, 64, 64, 3)
        """
        batch = super(GanDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        return list(zip(self.images))

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.images, = np.array(unpacked)

    def _n_samples(self):
        return len(self.images)


class CGanDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(CGanDataLoader, self).__init__(config)
        self.data_dir1 = config['data_loader']['data_dir1']
        self.data_dir2 = config['data_loader']['data_dir2']
        self.tags1_path = os.path.join(self.data_dir1, '../tags_clean.csv')
        self.tags2_path = os.path.join(self.data_dir2, '../tags.csv')
        self.img_tags1, self.img_tags2 = {}, {}  # format: {
                                                 #           img_idx0: {'hair': ['red'], 'eyes': ['blue']}
                                                 #           img_idx1: {'hair': ['blonde'], 'eyes': ['yellow']}
                                                 #           ...
                                                 #         }
        self.embedder = None
        self.images = []
        self.encoded_tags = []
        self.__parse_features(os.path.join(config['trainer']['save_dir'],
                                           config['name'], 'embedder.pkl'))
        self.__parse_images()

    def __parse_features(self, embedder_path):
        hair_attr, eyes_attr = [], []
        hair_stoplist = ['damage', 'long', 'short', 'pubic']
        eyes_stoplist = ['bicolored', 'gray']
        with open(self.tags1_path) as f:
            for line in f:
                idx, tags = line.split(',')
                tags = tags.split('\t')
                idx = int(idx)
                self.img_tags1[idx] = {'hair': [], 'eyes': []}
                for i, t in enumerate(tags):
                    t = t.strip(':0123456789')
                    t = t.split()
                    if 'hair' in t:
                        if t[0] not in hair_stoplist:
                            self.img_tags1[idx]['hair'].append(t[0])
                            if t[0] not in hair_attr:
                                hair_attr.append(t[0])

                    if 'eyes' in t and t[0] != 'eyes':
                        if t[0] not in eyes_stoplist:
                            self.img_tags1[idx]['eyes'].append(t[0])
                            if t[0] not in eyes_attr:
                                eyes_attr.append(t[0])

                if len(self.img_tags1[idx]['hair']) != 1 or len(self.img_tags1[idx]['eyes']) != 1:
                    self.img_tags1.pop(idx)
                    
        with open(self.tags2_path) as f:
            for line in f:
                idx, tags = line.split(',')
                tags = tags.split()
                idx = int(idx)
                if tags[0] not in hair_attr:
                    hair_attr.append(tags[0])
                if tags[2] not in eyes_attr:
                    eyes_attr.append(tags[2])
                self.img_tags2[idx] = {'hair': [tags[0]], 'eyes': [tags[2]]}
        
        hair_attr.sort()
        eyes_attr.sort()
        self.embedder = Embedder(hair_attr, eyes_attr)
        ensure_dir(os.path.dirname(embedder_path))
        pickle.dump(self.embedder, open(embedder_path, 'wb'))

    def __parse_images(self):
        # add images in data_dir1
        for img_file in sorted(os.listdir(self.data_dir1), key=lambda x: int(x[:x.rfind('.')])):
            img_idx = int(img_file[:img_file.rfind('.')])
            if img_idx not in self.img_tags1.keys(): continue
            img_path = os.path.join(self.data_dir1, img_file)
            img = cv2.imread(img_path)
            if img.shape != (64, 64, 3):
                img = cv2.resize(img, (64, 64))
            self.images.append(img)
            self.encoded_tags.append(self.embedder.encode_feature(self.img_tags1[img_idx]))
        # add images in data_dir2
        for img_file in sorted(os.listdir(self.data_dir2), key=lambda x: int(x[:x.rfind('.')])):
            img_idx = int(img_file[:img_file.rfind('.')])
            if img_idx not in self.img_tags2.keys(): continue
            img_path = os.path.join(self.data_dir2, img_file)
            img = cv2.imread(img_path)
            if img.shape != (64, 64, 3):
                img = cv2.resize(img, (64, 64))
            self.images.append(img)
            self.encoded_tags.append(self.embedder.encode_feature(self.img_tags2[img_idx]))
        self.images = (np.array(self.images) - 127.5)/127.5
        self.encoded_tags = np.array(self.encoded_tags)

    def __iter__(self):
        super(CGanDataLoader, self).__iter__()
        return self

    def __next__(self):
        """
        :return: img
            shape = (batch_size, 64, 64, 3)
        """
        batch = super(CGanDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        return list(zip(self.images, self.encoded_tags))

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [np.array(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.images, self.encoded_tags = unpacked

    def _n_samples(self):
        return len(self.images)


if __name__ == '__main__':
    import json
    test = CGanDataLoader(json.load(open('config.json')))
    print(test.images.shape, test.encoded_tags.shape)
    i = iter(test)
    a,b = next(i)
    print(a.shape, b.shape)
