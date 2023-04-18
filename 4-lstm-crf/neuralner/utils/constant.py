DEFAULT_TYPE = 'O'
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]
PAD_TYPE = '<PAD>'
PAD_TYPE_ID = 0
TYPE_TO_ID_IOB = {
    PAD_TYPE: PAD_TYPE_ID,
    DEFAULT_TYPE: 1,
    'B-LOC': 2,
    'B-MISC': 3,
    'B-ORG': 4,
    'B-PER': 5,
    'I-LOC': 6,
    'I-MISC': 7,
    'I-ORG': 8,
    'I-PER': 9
}
