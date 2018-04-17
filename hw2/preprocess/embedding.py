import numpy as np
from gensim.models.word2vec import Word2Vec

# FIXME: all embedders inherited from BaseEmbedder
# FIXME: pass *args, **kwargs to embedders


class BaseEmbedder:
    def __init__(self):
        pass

    def encode_word(self, word):
        pass

    def encode_line(self, line):
        pass

    def encode_lines(self, lines):
        pass

    def decode_word(self, word):
        pass

    def decode_line(self, line):
        pass

    def decode_lines(self, lines):
        pass

    def dec_out2dec_in(self, dec_out):
        pass


class Word2VecEmbedder:
    def __init__(self, corpus, emb_size):
        self.word_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.dictionary = dict((word, i) for i, word in enumerate(self.word_list))
        self.frequency = dict()
        self.emb_size = emb_size
        self.corpus = []
        for line in corpus:
            line = line.replace('.', '').split()
            line = [word.lower() for word in line]
            line = ['<BOS>'] + line
            self.corpus.append(line)
            for word in line:
                if word not in self.frequency:
                    self.frequency[word] = 1
                else:
                    self.frequency[word] += 1
        min_count = 4
        max_len = 0
        for i, line in enumerate(self.corpus):
            max_len = max(len(line), max_len)
            for j, word in enumerate(line):
                if self.frequency[word] < min_count:
                    self.corpus[i][j] = '<UNK>'
        for i, line in enumerate(self.corpus):
            self.corpus[i].append('<EOS>')
            self.corpus[i].extend(['<PAD>'] * (max_len-len(self.corpus[i])))

        self.word2vec = Word2Vec.load('w2v.pkl')
        # self.word2vec = Word2Vec(self.corpus, size=self.emb_size, min_count=min_count, iter=30, workers=16)
        # self.word2vec.save('w2v.pkl')

    def encode_word(self, word):
        if word in self.word2vec.wv:
            return self.word2vec.wv[word]
        else:
            return self.word2vec.wv['<UNK>']

    def encode_line(self, line):
        encoded = []
        for word in line:
            encoded.append(self.encode_word(word))
        return encoded

    def encode_lines(self, lines):
        encoded = []
        for line in lines:
            line = line.replace('.', '').split()
            line = [word.lower() for word in line]
            encoded.append(np.array(self.encode_line(line)))
        return encoded

    def decode_lines(self, lines):
        decoded = []
        for line in lines:
            line = [self.word2vec.wv.most_similar(positive=[vec], topn=1)[0][0] for vec in line]
            line = ' '.join(line)
            line = line.split('<EOS>', 1)[0]
            line = line.split('<PAD>', 1)[0]
            line = line.split()
            if len(line) == 0:
                line = ['a']
            line = ' '.join(line)
            decoded.append(line)
        return decoded

    def dec_out2dec_in(self, dec_out):
        return dec_out.cpu().data.numpy()


class OneHotEmbedder:
    def __init__(self, corpus, emb_size):
        self.word_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.dictionary = dict((word, i) for i, word in enumerate(self.word_list))
        self.frequency = dict()
        self.dict_size = emb_size
        for line in corpus:
            line = line.replace('.', '').split()
            line = [word.lower() for word in line]
            for word in line:
                if word not in self.frequency:
                    self.frequency[word] = 1
                else:
                    self.frequency[word] += 1
        frequency_sorted = sorted(self.frequency, key=self.frequency.get, reverse=True)
        token_count = len(self.dictionary)
        for word in frequency_sorted[:emb_size - token_count]:
            self.dictionary[word] = len(self.dictionary)
            self.word_list.append(word)
        assert len(self.word_list) == emb_size

    def encode_word(self, word):
        return onehot(self.dict_size, self.dictionary[word])

    def encode_lines(self, lines):
        encoded = []
        for line in lines:
            line = line.replace('.', '').split()
            line = [word.lower() for word in line]
            line = [onehot(self.dict_size,
                           self.dictionary.get(word, self.dictionary['<UNK>'])) for word in line]
            # line = [self.dictionary.get(word, self.dictionary['<UNK>']) for word in line]
            encoded.append(np.array(line))
        return encoded

    def decode_lines(self, lines):
        decoded = []
        for line in lines:
            line = [self.word_list[int(np.argmax(vec, 0))] for vec in line]
            line = ' '.join(line)
            line = line.split('<EOS>', 1)[0]
            line = line.split('<PAD>', 1)[0]
            line = line.split()
            if len(line) == 0:
                line = ['a']
            line = ' '.join(line)
            decoded.append(line)
        return decoded

    def dec_out2dec_in(self, dec_out):
        dec_in = []
        dec_out_batch = dec_out.cpu().data.numpy()[0]
        dec_out_batch = np.argmax(dec_out_batch, 1)
        for dec_out in dec_out_batch:
            dec_in.append(onehot(self.dict_size, dec_out))
        return np.array([dec_in])


def onehot(dim, label):
    v = np.zeros((dim,))
    v[label] = 1
    return v


if __name__ == '__main__':
    embedder = OneHotEmbedder(['i am sleeping.', 'she is such a good'], emb_size=12)

    enc = embedder.encode_lines(['i is such', 'a a am sleeping good'])
    dec = embedder.decode_lines(enc)

    print(dec)
