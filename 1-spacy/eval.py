import argparse
import re
import spacy
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

    def _parse_sentence(self, raw_sentence):
        return [
            tuple(token.split(self._sep))
            for token in raw_sentence.strip().split("\n")
        ]

    @classmethod
    def token_to_iob(self, token):
        iob = token.ent_iob_
        return (
            token.ent_iob_ if token.ent_iob_ == 'O'
            else f'{token.ent_iob_}-{token.ent_type_}'
        )


def parse_args():
    description = "Eval results from a NER"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='test dataset to be evaluated',
    )
    parser.add_argument(
        '-m',
        '--model',
        default='model-best',
        type=str,
        metavar='MODEL',
        help='model to be evaluated'
    )

    return parser.parse_args()


def flatten(y):
    return list(chain.from_iterable(y))


args = parse_args()

nlp = spacy.load(args.model)

sents = [
    [token for token in sent] for sent in IOB(args.dataset).convert_file()
]
golden = [[token[-1] for token in sent] for sent in sents]
tokens = [[token[0] for token in sent] for sent in sents]

predict = []
for sent in tqdm(tokens):
    parsed = nlp(spacy.tokens.Doc(nlp.vocab, sent))
    predict.append([
        IOB.token_to_iob(token)
        for token in nlp(spacy.tokens.Doc(nlp.vocab, sent))
    ])

labels = sorted(
    [label for label in TAGS if label != 'O'],
    key=lambda name: (name[1:], name[0]))

results = classification_report(
    flatten(golden), flatten(predict), labels=labels, digits=3)

print(results)
print()

results = seq_classification_report(golden, predict, digits=3)
print(results)
