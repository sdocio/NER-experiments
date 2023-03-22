import argparse
import sys
import warnings
from itertools import chain
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report

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


class Predictions:
    def __init__(self, ifile):
        self._ifile = ifile

    def convert_file(self):
        return [
            [tag for tag in sent.split()]
            for sent in self._read_sentences_from_file()
        ]

    def _read_sentences_from_file(self):
        try:
            with open(self._ifile) as fhi:
                for line in fhi:
                    yield line.strip()
        except IOError as err:
            print(err, file=sys.stderr)
            sys.exit()


def parse_args():
    description = "Eval results from a NER"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='dataset to be evaluated',
    )
    parser.add_argument(
        'golden',
        type=str,
        help='dataset with gold standard',
    )
    parser.add_argument(
        '-s',
        '--strict',
        action='store_true',
        default=False,
        help='stop eval if datasets have inconsistencies',
    )

    return parser.parse_args()


def check(data1, data2):
    assert len(data1) == len(data2), (
        "Different number of sentences between datasets"
    )
    for i in range(len(data1)):
        assert len(data1[i]) == len(data2[i]), (
            f"Different number of tokens in line {i+1}")


def flatten(y):
    return list(chain.from_iterable(y))


def fix(data1, data2):
    skip = 0
    for i in range(len(data1)-1, -1, -1):
        if len(data1[i]) != len(data2[i]):
            del data1[i]
            del data2[i]
            skip += 1
    print(f"Skipped {skip} inconsistent sentences.", file=sys.stderr)


args = parse_args()

golden_tags = [
    [token[-1] for token in sent] for sent in IOB(args.golden).convert_file()
]
dataset_tags = Predictions(args.dataset).convert_file()

if args.strict is False:
    fix(golden_tags, dataset_tags)
check(golden_tags, dataset_tags)

labels = sorted(
    [label for label in TAGS if label != 'O'],
    key=lambda name: (name[1:], name[0]))

results = classification_report(
    flatten(golden_tags), flatten(dataset_tags), labels=labels, digits=3)

print(results)
print()

results = seq_classification_report(golden_tags, dataset_tags, digits=3)
print(results)
