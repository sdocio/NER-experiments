"""
Script to perform kfold cross validation with a CRF model.
"""
import argparse
import csv
import sys
from collections import defaultdict
from operator import itemgetter
from pprint import PrettyPrinter
from statistics import pstdev, mean
import sklearn_crfsuite
from sklearn.model_selection import KFold
from seqeval.metrics import classification_report as seq_classification_report
from IOB import IOB
from features import CRFFeatures
from config import (
    version, shuffle, random_state, c1, c2, max_iterations,
    all_possible_transitions
)


def parse_args():
    """
    Manage command-line arguments
    """
    description = "Script to perform kfold cross validation with a CRF model."

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-V',
        '--version',
        action='version',
        version=f'%(prog)s {version}'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='verbose'
    )
    parser.add_argument(
        '-p',
        '--with-pos',
        action='store_true',
        default=False,
        help='use POS tags as feature'
    )
    parser.add_argument(
        '-k',
        '--kfolds',
        default='10',
        type=int,
        help='number of folds'
    )
    parser.add_argument(
        '-f',
        '--full',
        action='store_true',
        default=False,
        help='show full results'
    )
    parser.add_argument(
        'dataset',
        metavar='input file',
        type=str,
        help='dataset file'
    )
    return parser.parse_args()


def print_csv(results: dict, full_info=False) -> None:
    """
    Print results in csv format
    """
    ofile = csv.writer(
        sys.stdout,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL)

    if full_info:
        ofile.writerow([
            'label',
            'metric',
            *[f'iter{n}' for n in range(args.kfolds)]
        ])
        for dtype in results:
            for dmetric in results[dtype]:
                ofile.writerow([
                    dtype, dmetric, *results[dtype][dmetric]
                ])
    else:
        ofile.writerow(['metric', 'mean', 'stdev'])
        for metric in ["precision", "recall", "f1-score"]:
            mean_v = mean(list(results["micro avg"][metric]))
            stdev_v = pstdev(list(results["micro avg"][metric]))
            ofile.writerow([metric, mean_v, stdev_v])


def update_results(results: dict) -> None:
    """
    Update dictionary containing results with the current iteration
    """
    for value in [
        "LOC", "MISC", "ORG", "PER", "micro avg", "macro avg",
        "weighted avg"
    ]:
        for score in ["precision", "recall", "f1-score"]:
            full_results[value][score].append(results[value][score])


args = parse_args()
iob = IOB()
feats = CRFFeatures(with_pos=args.with_pos)

sentences = iob.parse_file(args.dataset)
sentences = [
        [(token[0], token[1]) for token in sent]
        for sent in iob.parse_file(args.dataset)
    ]

X = [feats.sent2features(s) for s in sentences]
y = [feats.sent2labels(s) for s in sentences]

ITER = 0
full_results = defaultdict(lambda: defaultdict(list))

if shuffle is False:
    random_state = None

for train, test in KFold(
    n_splits=args.kfolds,
    shuffle=shuffle,
    random_state=random_state
).split(X):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions,
        verbose=False,
    )

    X_train = list(itemgetter(*train)(X))
    y_train = list(itemgetter(*train)(y))
    X_test = list(itemgetter(*test)(X))
    y_test = list(itemgetter(*test)(y))

    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)

    iter_results = seq_classification_report(
        y_test,
        y_pred,
        digits=3,
        output_dict=True)

    if args.verbose:
        pp = PrettyPrinter(stream=sys.stderr)
        print("==================", file=sys.stderr)
        print(f"iteration {ITER}", file=sys.stderr)
        print("==================", file=sys.stderr)
        print(f"Train length: {len(train)}", file=sys.stderr)
        print(f"Test length: {len(test)}", file=sys.stderr)
        pp.pprint(iter_results)
        print(file=sys.stderr)

    update_results(iter_results)
    ITER += 1

print_csv(full_results, full_info=args.full)
