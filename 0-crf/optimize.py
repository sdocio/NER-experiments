import argparse
import pickle
from itertools import chain
from scipy import stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn_crfsuite import metrics, CRF
from sklearn.metrics import make_scorer
from IOB import IOB
from features import CRFFeatures
from config import version, test_size, random_state, shuffle


def parse_args():
    description = "Use RandomizedSearchCV to optimize hyperparameters."

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
        'dataset',
        metavar='input file',
        type=str,
        help='dataset file'
    )
    return parser.parse_args()


def flatten(y):
    return list(chain.from_iterable(y))


args = parse_args()
iob = IOB()

crf = CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True,
    verbose=args.verbose,
)
feats = CRFFeatures(with_pos=args.with_pos)

sentences = iob.parse_file(args.dataset)
train, test = train_test_split(
    sentences,
    test_size=test_size,
    random_state=random_state,
    shuffle=shuffle
)

X_train = [feats.sent2features(s) for s in train]
y_train = [feats.sent2labels(s) for s in train]

sorted_labels = sorted(
    [label for label in set(flatten(y_train)) if label != 'O'],
    key=lambda name: (name[1:], name[0]))

param_grid = {
    'c1': stats.expon(scale=0.5),
    'c2': stats.expon(scale=0.05),
}
f1_scorer = make_scorer(
    metrics.flat_f1_score,
    average='weighted',
    labels=sorted_labels)

rs = RandomizedSearchCV(estimator=crf,
                        param_distributions=param_grid,
                        scoring=f1_scorer,
                        cv=3,
                        verbose=True,
                        n_iter=100,
                        random_state=random_state,
                        n_jobs=-1)

rs.fit(X_train, y_train)

for k, v in rs.best_estimator_.get_params().items():
    print('{}: {}'.format(k, v))
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)

crf = pickle.dump(rs.best_estimator_, open("crf-best_estimator.model", 'wb'))
