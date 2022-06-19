import argparse
import warnings
import sys
import sklearn_crfsuite
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn_crfsuite.metrics import sequence_accuracy_score
from seqeval.metrics import classification_report as seq_classification_report
from itertools import chain
from IOB import IOB


__version = '0.0.2'


def parse_args():
    description = "Test a CRF model for NER using IOB2 datasets."

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'%(prog)s {__version}'
    )
    parser.add_argument(
        '-m',
        '--model',
        default='crf.model',
        type=str,
        metavar='FILE',
        help='model file'
    )
    parser.add_argument(
        'dataset',
        metavar='input file',
        type=str,
        help='Corpus file'
    )
    return parser.parse_args()


def flatten(y):
    return list(chain.from_iterable(y))


args = parse_args()
iob = IOB()
crf = pickle.load(open(args.model, 'rb'))

sentences = iob.parse_file(args.dataset)
train, test = train_test_split(
    sentences,
    test_size=crf.test_size,
    random_state=crf.random_state,
    shuffle=crf.shuffle
)

X_test = [crf.sent2features(s) for s in test]
y_test = [crf.sent2labels(s) for s in test]

# predict
y_pred = crf.predict(X_test)

labels = sorted(
    [label for label in crf.classes_ if label != 'O'],
    key=lambda name: (name[1:], name[0]))
results = classification_report(
    flatten(y_test), flatten(y_pred), labels=labels, digits=3)
accuracy = crf.score(X_test, y_test)

print(results)
print()
print(f"Accuracy: {accuracy}")

print()

results = seq_classification_report(y_test, y_pred)
print(results)
# print(
# flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

# scores = cross_val_score(clf, X, y, cv=5)
