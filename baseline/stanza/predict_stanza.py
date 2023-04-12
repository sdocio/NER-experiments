import argparse
import stanza
import sys
import warnings
from itertools import chain
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")
TAGS = [
    'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER'
]


class IOB:
    def __init__(self, ifile, sep=" "):
        self._ifile = ifile
        self._sep = sep

    def convert_file(self):
        return [
            self._parse_sentence(raw)
            for raw in self._read_sentences_from_file(self._ifile)
        ]

    def _parse_sentence(self, raw_sentence):
        return [
            tuple(token.split(self._sep))
            for token in raw_sentence.strip().split("\n")
        ]

    def _read_sentences_from_file(self, ifile):
        raw_sentence = ""
        try:
            with open(ifile) as fhi:
                for line in fhi:
                    if line == "\n":
                        if raw_sentence == "":
                            continue
                        yield raw_sentence
                        raw_sentence = ""
                        continue

                    if line:
                        raw_sentence += line

            if raw_sentence:
                yield raw_sentence
        except IOError as err:
            print(err, file=sys.stderr)
            sys.exit()


class Stanza:
    def __init__(self, separator=" ", mapping={}, model=None, segment=False):
        self._sep = separator
        self._mapping = mapping
        self._segment = segment
        self._model = model
        self.store_pos = False

        try:
            self.nlp = stanza.Pipeline(
                lang='es',
                processors={'ner': self._model},
                tokenize_pretokenized=True,
                dir='models',
                ner_batch_size=32,
                logging_level='WARN'
            )
        except Exception as err:
            print(f'Unable to load stanza model:\n{err}', file=sys.stderr)
            sys.exit(1)

    def __call__(self, docs):
        return self.nlp(docs).iter_tokens()

    def tokenizer(self, text):
        return [t.text for t in self.nlp(text).iter_tokens()]

    def token_to_iob(self, token):
        iob = token.ner
        if '-' in iob:
            iob, ent_type_ = token.ner.split('-')
            enttype = self._mapping.get(ent_type_, ent_type_)
            iob = 'B' if iob == 'S' or iob == 'B' else 'I'
        iob_tag = iob if iob == 'O' else f'{iob}-{enttype}'

        if self.store_pos:
            return self._sep.join([token.text, token.upos, iob_tag])
        return iob_tag


def parse_args():
    description = "Predict and evaluate NER tags using a Stanza model"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='IOB input file',
    )
    parser.add_argument(
        '-m',
        '--model',
        default='conll02',
        type=str,
        metavar='MODEL',
        help='model'
    )
    parser.add_argument(
        '-e',
        '--eval',
        action='store_true',
        default=False,
        help='Evaluation only'
    )

    return parser.parse_args()


def flatten(y):
    return list(chain.from_iterable(y))


args = parse_args()
nlp = Stanza(model=args.model)

sentences = [
    [token for token in sent] for sent in IOB(args.dataset).convert_file()
]
golden = [[token[-1] for token in sent] for sent in sentences]
labels = sorted(
    [label for label in TAGS if label != 'O'],
    key=lambda name: (name[1:], name[0]))

predict = []
for sent in tqdm(sentences):
    tokens = [token[0] for token in sent]
    predict.append([
        [tokens[i], nlp.token_to_iob(token)]
        for i, token in enumerate(nlp([tokens]))
    ])
y_pred = [[token[-1] for token in sent] for sent in predict]

if not args.eval:
    for sent in predict:
        for token in sent:
            print(" ".join(token))
        print()

results = classification_report(
    flatten(golden), flatten(y_pred), labels=labels, digits=3)

print(results, file=sys.stderr)
print()

results = seq_classification_report(golden, y_pred, digits=3)
print(results, file=sys.stderr)