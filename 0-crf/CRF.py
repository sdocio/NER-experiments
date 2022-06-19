import sklearn_crfsuite

pos_map = {
    'ADJ': 'A',
    'ADP': 'S',
    'ADV': 'R',
    'AUX': 'V',
    'CCONJ': 'C',
    'DET': 'D',
    'INTJ': 'I',
    'NOUN': 'N',
    'NUM': 'Z',
    'PART': 'Y',
    'PRON': 'P',
    'PROPN': 'E',
    'PUNCT': 'F',
    'SCONJ': 'B',
    'SYM': 'X',
    'VERB': 'V',
    'X': 'X',
}


class CRF(sklearn_crfsuite.CRF):
    test_size = 0.2
    random_state = None
    shuffle = True

    def __init__(self, with_pos, **kwargs):
        super().__init__(**kwargs)
        self.with_pos = with_pos

    def word2features(self, sent, i):
        word = sent[i][0]

        if self.with_pos and len(sent[i]) < 2:
            raise Exception(
                "With PoS option, the dataset must have two fields.")

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }

        if(self.with_pos):
            postag = pos_map.get(sent[i][1], 'X')
            features.update({'postag': postag})

        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })

            if self.with_pos:
                features.update({'-1:postag': pos_map.get(sent[i-1][1], 'X')})
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })

            if self.with_pos:
                features.update({'+1:postag': pos_map.get(sent[i+1][1], 'X')})
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [token[-1] for token in sent]
