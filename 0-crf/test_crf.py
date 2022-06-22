import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report
from itertools import chain
from IOB import IOB
from features import CRFFeatures
from config import version, test_size, random_state, shuffle


def parse_args():
    description = "Test a CRF model for NER using IOB2 datasets."

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
        '-p',
        '--with-pos',
        action='store_true',
        default=False,
        help='use POS tags as feature'
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
feats = CRFFeatures(with_pos=args.with_pos)
crf = pickle.load(open(args.model, 'rb'))

sentences = iob.parse_file(args.dataset)
train, test = train_test_split(
    sentences,
    test_size=test_size,
    random_state=random_state,
    shuffle=shuffle
)

X_test = [feats.sent2features(s) for s in test]
y_test = [feats.sent2labels(s) for s in test]

y_pred = crf.predict(X_test)

labels = sorted(
    [label for label in crf.classes_ if label != 'O'],
    key=lambda name: (name[1:], name[0]))
results = classification_report(
    flatten(y_test), flatten(y_pred), labels=labels, digits=3)
accuracy = crf.score(X_test, y_test)

print(results)
print()

results = seq_classification_report(y_test, y_pred, digits=3)
print(results)
