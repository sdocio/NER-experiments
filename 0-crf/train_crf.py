import argparse
import pickle
import sklearn_crfsuite
from itertools import chain
from sklearn.model_selection import train_test_split
from IOB import IOB
from features import CRFFeatures
from config import version, test_size, random_state, shuffle


def parse_args():
    description = "Train a CRF model for NER from IOB2 datasets."

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
        '-o',
        '--output',
        default='crf.model',
        type=str,
        metavar='FILE',
        help='model output file',
    )
    parser.add_argument(
        'dataset',
        metavar='input file',
        type=str,
        help='dataset file'
    )
    return parser.parse_args()


args = parse_args()
iob = IOB()
feats = CRFFeatures(with_pos=args.with_pos)
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=args.verbose,
)

sentences = iob.parse_file(args.dataset)
train, test = train_test_split(
    sentences,
    test_size=test_size,
    random_state=random_state,
    shuffle=shuffle
)

if args.verbose:
    train_size = len(list(chain(*train)))
    test_size = len(list(chain(*test)))
    total = train_size + test_size
    print(
        "train size (in tokens): ",
        train_size, "(%2.2f%%)" % ((train_size*100)/total))
    print(
        "train size (in sentences): ",
        len(train), "(%2.2f%%)" % ((len(train)*100)/len(sentences)))
    print(
        "test size (in tokens): ",
        test_size, "(%2.2f%%)" % ((test_size*100)/total))
    print(
        "test size (in sentences): ",
        len(test), "(%2.2f%%)" % ((len(test)*100)/len(sentences)))
    print("random_state: ", random_state)
    print("shuffle: ", shuffle)
    print()

X_train = [feats.sent2features(s) for s in train]
y_train = [feats.sent2labels(s) for s in train]

crf.fit(X_train, y_train)
pickle.dump(crf, open(args.output, 'wb'))
