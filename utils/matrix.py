"""
Compares two IOB2 datasets with NER tags (LOC, PER, ORG, MISC) and produces
a confusion matrix.
"""
import argparse
import sys
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


TAGS = [
    'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER'
]


class IOB:
    """
    Class to manage IOB files
    """
    def __init__(self, ifile, sep=" "):
        self._ifile = ifile
        self._sep = sep

    def convert_file(self):
        '''
        Takes a filename and returns a nested list with tuples containing
        the tokens.

        >>> iob.convert_file("../datasets/test.iob")
        [[('Não', 'O'), ('sei', 'O'), ('.', 'O')], [('Não', 'O'), ('.', 'O')]]
        '''
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


def parse_args():
    """Parses script arguments"""
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
        '-o',
        '--output',
        type=str,
        help='output file',
        default='cm.pdf',
    )

    return parser.parse_args()


def check(data1, data2):
    """
    Takes two nested lists and cheks if their dimensions are the same
    """
    assert len(data1) == len(data2), (
        "Different number of sentences between datasets"
    )
    for i in range(len(data1)):
        assert len(data1[i]) == len(data2[i]), (
            f"Different number of tokens in line {i+1}")


def flatten(nested_list):
    """
    Flattens a nested list.
    """
    return list(chain.from_iterable(nested_list))


args = parse_args()
golden_tags = [
    [token[-1] for token in sent] for sent in IOB(args.golden).convert_file()
]
dataset_tags = [
    [token[-1] for token in sent] for sent in IOB(args.dataset).convert_file()
]

check(golden_tags, dataset_tags)

cm = confusion_matrix(
    flatten(golden_tags),
    flatten(dataset_tags),
    normalize="true")
np.set_printoptions(precision=3, suppress=True)

plt.figure(figsize=(8, 5))
fx = sns.heatmap(cm, annot=True, fmt=".3f", cmap="GnBu")
fx.set_title('Confusion Matrix')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values')
fx.xaxis.set_ticklabels(TAGS + ["O"])
fx.yaxis.set_ticklabels(TAGS + ["O"])
fx.tick_params(axis="y", rotation=20)
plt.savefig(args.output, format="pdf", bbox_inches="tight")
