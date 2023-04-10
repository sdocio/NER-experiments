import argparse
import csv
import sys
import warnings
from itertools import chain
from seqeval.metrics import classification_report as seq_classification_report

warnings.filterwarnings("ignore")


class IOB:
    def __init__(self, sep=" "):
        self._sep = sep

    def parse_file(self, ifile):
        return [
            self._parse_sentence(raw)
            for raw in self._read_sentences_from_file(ifile)
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
        except IOError:
            print("Unable to read file: " + ifile)
            sys.exit()


def parse_args():
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
        '-c',
        '--csv',
        action='store_true',
        default=False,
        help='resulting metrics are converted into csv',
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


def convert_csv(results):
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

if args.csv:
    results = seq_classification_report(
        golden_tags,
        dataset_tags,
        digits=3,
        output_dict=True)
    convert_csv(results)
else:
    results = seq_classification_report(
        golden_tags,
        dataset_tags,
        digits=3)
    print(results)
