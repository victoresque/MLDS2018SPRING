import torch, cv2, os
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader


class GanDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(GanDataLoader, self).__init__(config)
        self.data_dir1 = config['data_loader']['data_dir1']
        self.data_dir2 = config['data_loader']['data_dir2']
        self.noise_dim = config['data_loader']['noise_dim']
        self.noise = None
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
        self.images = self.images[:1000]

    def __iter__(self):
        self.noise = None
        super(GanDataLoader, self).__iter__()
        n_img = len(self.images)
        self.noise = [np.random.normal(0, 1, (self.noise_dim,)) for _ in range(n_img)]
        return self

    def __next__(self):
        """
        :return: noise, img, label(zero first, one follow)
            shape = (batch_size, noise_dim), (batch_size, 64,64,3), (2*batch_size,)
        """
        batch = super(GanDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]  # [[noise in batch], [img in batch]]
        label = np.concatenate((np.zeros(len(batch[0])), np.ones(len(batch[0]))))
        batch.append(label)
        return batch

    def _pack_data(self):
        if self.noise is not None:
            packed = list(zip(self.noise, self.images))
        else:
            packed = [(img,) for img in self.images]
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.images, = np.array(unpacked)

    def _n_samples(self):
        return len(self.images)


if __name__ == '__main__':
    import json
    test = GanDataLoader(json.load(open('config.json')))
    i = iter(test)
    print(test.images.shape)
    a,b,c = next(i)
    print(a.shape, b.shape, c.shape)
