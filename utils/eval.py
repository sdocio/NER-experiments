"""
Script that evaluates a IOB dataset using a gold standard
"""
import argparse
import csv
import sys
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix

TAGS = [
    'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER'
]


class IOB:
    """
    Class to manage IOB files
    """
    def __init__(self, sep: str = " "):
        self._sep = sep

    def __repr__(self):
        return f'IOB object with separation char = "{self._sep}"'

    def parse_file(self, ifile: str) -> list:
        '''
        Takes a filename and returns a nested list with tuples containing
        the tokens.

        >>> iob.parse_file("../datasets/test.iob")
        [[('Não', 'O'), ('sei', 'O'), ('.', 'O')], [('Não', 'O'), ('.', 'O')]]
        '''
        return [
            self._parse_sentence(raw)
            for raw in self._read_sentences_from_file(ifile)
        ]

    def _parse_sentence(self, raw_sentence: str) -> list:
        return [
            (token.split(self._sep))
            for token in raw_sentence.strip().split("\n")
        ]

    def _read_sentences_from_file(self, ifile: str):
        raw_sentence = ""
        try:
            with open(ifile, encoding="utf-8") as fhi:
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
        except IOError:
            print("Unable to read file: " + ifile)
            sys.exit()


def parse_args() -> argparse.Namespace:
    """Parses script arguments"""
    description = "Eval model results comparing two IOB files."

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
    parser.add_argument(
        '-o',
        '--output',
        choices=['csv', 'plot', 'text'],
        default='text',
        help='resulting metrics are converted into csv',
    )
    parser.add_argument(
        '-p',
        '--plot-file',
        nargs=1,
        default='cm.pdf',
        help='filename to store the plot',
    )

    return parser.parse_args()


def check(data1: list, data2: list) -> None:
    """
    Takes two nested lists and cheks if their dimensions are the same
    """
    assert len(data1) == len(data2), (
        "Different number of sentences between datasets"
    )
    for i, _ in enumerate(data1):
        assert len(data1[i]) == len(data2[i]), (
            f"Different number of tokens in line {i+1}")


def fix(data1, data2) -> None:
    """Removes inconsistent entries (with different lengths) in the datasets"""
    skip = 0
    try:
        for i in range(len(data1)-1, -1, -1):
            if len(data1[i]) != len(data2[i]):
                del data1[i]
                del data2[i]
                skip += 1
    except IndexError as exc:
        raise IndexError("datasets with different sizes") from exc
    if skip > 0:
        print(f"Skipped {skip} inconsistent sentences.", file=sys.stderr)


def flatten(nested_dataset: list) -> list:
    """Takes a nested list and returns it unnested"""
    return list(chain.from_iterable(nested_dataset))


def convert_csv(results: dict) -> None:
    """
    Writes a CSV file using the dictionary returned by
    seqeval.classification_report() with the evaluation results
    """
    ofile = csv.writer(
        sys.stdout,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL)

    for label in ["LOC", "MISC", "ORG", "PER"]:
        ofile.writerow([
            label,
            f'{results[label]["precision"]:.3f}',
            f'{results[label]["recall"]:.3f}',
            f'{results[label]["f1-score"]:.3f}',
            ])
    for metric in ["micro avg", "macro avg", "weighted avg"]:
        ofile.writerow([
            metric,
            f'{results[metric]["precision"]:.3f}',
            f'{results[metric]["recall"]:.3f}',
            f'{results[metric]["f1-score"]:.3f}',
            ])


def evaluate() -> None:
    """
    Takes a dataset in IOB2 format and evaluates it against a gold standard
    """
    args = parse_args()
    iob = IOB()

    golden_tags = [
        [token[-1] for token in sent] for sent in iob.parse_file(args.golden)
    ]
    dataset_tags = [
        [token[-1] for token in sent] for sent in iob.parse_file(args.dataset)
    ]

    if args.strict is False:
        fix(golden_tags, dataset_tags)
    check(golden_tags, dataset_tags)

    if args.output == 'csv':
        eval_results = classification_report(
            golden_tags,
            dataset_tags,
            digits=3,
            output_dict=True)
        convert_csv(eval_results)
    elif args.output == 'plot':
        cmatrix = confusion_matrix(
            flatten(golden_tags), flatten(dataset_tags), normalize="true")
        plt.figure(figsize=(8, 5))
        plot = sns.heatmap(cmatrix, annot=True, fmt=".3f", cmap="GnBu")
        plot.set_xlabel('Predicted Values')
        plot.set_ylabel('Actual Values')
        plot.xaxis.set_ticklabels(TAGS + ["O"])
        plot.yaxis.set_ticklabels(TAGS + ["O"])
        plot.tick_params(axis="y", rotation=20)
        plt.savefig(args.plot_file, format="pdf", bbox_inches="tight")
    else:
        eval_results = classification_report(
            golden_tags,
            dataset_tags,
            digits=3)
        print(eval_results)


if __name__ == "__main__":
    evaluate()
