import os
import pickle


class Vocab(object):
    def __init__(self, filename, word_counter=None, threshold=0):
        assert os.path.exists(filename), "Vocab file does not exist: " + filename  # noqa
        self.id2word, self.word2id = self.load(filename)
        self.size = len(self.id2word)

    def load(self, filename):
        with open(filename, 'rb') as infile:
            id2word = pickle.load(infile)
            word2id = dict(
                [(id2word[idx], idx) for idx in range(len(id2word))]
            )
        return id2word, word2id
