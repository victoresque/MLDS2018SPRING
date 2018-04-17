from base import BaseDataLoader


class ChatbotDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, embedder, emb_size, shuffle=True, mode='train'):
        shuffle = shuffle if mode == 'train' else False
        super(ChatbotDataLoader, self).__init__(batch_size, shuffle)
        pass
