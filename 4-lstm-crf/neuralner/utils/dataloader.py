import torch
from neuralner.utils import constant
from neuralner.utils.iob import IOB

INPUT_SIZE = 3


class DataLoader(object):
    def __init__(
            self,
            filename,
            opt,
            vocab,
            char_vocab):
        self.opt = opt
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.label2id = constant.TYPE_TO_ID_IOB

        batch_size = opt['batch_size']
        iob = IOB()
        data = [
            sent.update(
                {'char': [list(tok) for tok in sent['token']]}
            ) or sent
            for sent in iob.parse_file(filename)
        ]

        self.raw_data = data
        data = self.preprocess(data, opt)
        self.id2label = dict([(v, k) for k, v in self.label2id.items()])
        self.labels = [[self.id2label[lid] for lid in d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def preprocess(self, data, opt):
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            tokens = map_to_ids(tokens, self.vocab.word2id)
            chars = [map_to_ids(c, self.char_vocab.word2id) for c in d['char']]
            types = map_to_ids(d['tag'], self.label2id)
            processed += [[tokens, chars, types]]
        return processed

    def gold(self):
        return self.labels

    def words(self):
        words = [d['token'] for d in self.raw_data]
        return words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == INPUT_SIZE

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, constant.PAD_ID)
        chars = get_long_tensor_3d(batch[1], batch_size)
        types = get_long_tensor(
            batch[2], batch_size, pad_id=constant.PAD_TYPE_ID)
        return (words, masks, chars, types, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_long_tensor(tokens_list, batch_size, pad_id=constant.PAD_ID):
    l = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, l).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def get_long_tensor_3d(tokens_list, batch_size, pad_id=constant.PAD_ID):
    l = max(len(ts) for ts in tokens_list)
    d = max(max([len(x) for x in ts]) for ts in tokens_list)
    tokens = torch.LongTensor(batch_size, l, d).fill_(pad_id)
    for i, ts in enumerate(tokens_list):
        for j, t in enumerate(ts):
            tokens[i, j, :len(t)] = torch.LongTensor(t)
    return tokens


def get_float_tensor(features_list, batch_size):
    if features_list is None or features_list[0] is None:
        return None
    seq_len = max(len(x) for x in features_list)
    feature_len = len(features_list[0][0])
    features = torch.FloatTensor(batch_size, seq_len, feature_len).zero_()
    for i, f in enumerate(features_list):
        features[i, :len(f), :] = torch.FloatTensor(f)
    return features


def sort_all(batch, lens):
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [
        list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))
    ]
    return sorted_all[2:], sorted_all[1]