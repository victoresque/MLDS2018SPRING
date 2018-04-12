import numpy as np


class OneHot:
    def __init__(self, corpus, dict_size=1000):
        self.dictionary = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
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

    def get_dictionary(self):
        return self.dictionary


def onehot(dim, label):
    v = np.zeros((dim,))
    v[label] = 1
    return v


if __name__ == '__main__':
    print(onehot(6, 2))
