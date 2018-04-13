import numpy as np


class OneHot:
    def __init__(self, corpus, dict_size=1000):
        self.word_list = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.dictionary = dict((word, i) for i, word in enumerate(self.word_list))
        self.frequency = dict()
        self.dict_size = 1000
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
        for word in frequency_sorted[:dict_size-token_count]:
            self.dictionary[word] = len(self.dictionary)
            self.word_list.append(word)

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
            decoded.append(line)
        return decoded


def onehot(dim, label):
    v = np.zeros((dim,))
    v[label] = 1
    return v


if __name__ == '__main__':
    print(onehot(6, 2))
