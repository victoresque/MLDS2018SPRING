import torch, cv2, os
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MnistDataLoader, self).__init__(config)
        self.data_dir1 = config['data_loader']['data_dir1']
        self.data_dir2 = config['data_loader']['data_dir2']
        self.noise_dim = config['data_loader']['noise_dim']
        self.noise = None
        self.images = []
        self.__parse_images(config['data_loader']['noise_seed'])
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __parse_images(self, seed):
        # add images in data_dir1
        for img_file in sorted(os.listdir(self.data_dir1), key=lambda x: int(x[:x.rfind('.')])):
            img_path = os.path.join(self.data_dir1, img_file)
            img = cv2.imread(img_path)
            if img.shpae != (64,64,3):
                img = cv2.resize(img, (64,64))
            self.images.append(img)
        # add images in data_dir2
        for img_file in sorted(os.listdir(self.data_dir2), key=lambda x: int(x[:x.rfind('.')])):
            img_path = os.path.join(self.data_dir2, img_file)
            img = cv2.imread(img_path)
            if img.shpae != (64,64,3):
                img = cv2.resize(img, (64,64))
            self.images.append(img)
        # generate noise
        n_img = len(self.images)
        np.random.seed(seed)
        self.noise = np.random.normal(0,1,(n_img, self.noise_dim))
        self.images = np.array(self.images)

    def __next__(self):
        """
        :return: noise, img, label
            shape = (batch_size, noise_dim), (batch_size, 64,64,3), (2*batch_size,)
        """
        batch = super(MnistDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch] # [[noise in batch], [img in batch]]
        label = np.concatenate(np.zeros(len(batch[0])), np.ones(len(batch[0])))
        batch.append(label)
        return batch

    def _pack_data(self):
        packed = list(zip(self.noise, self.images))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)
