import argparse
import pickle
import joblib


def parse_args():
    description = "Converts a pickle file into joblib"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'pickle',
        metavar='input file',
        type=str,
        help='pickle file'
    )
    parser.add_argument(
        'joblib',
        metavar='output file',
        type=str,
        help='joblib file'
    )

    return parser.parse_args()


args = parse_args()
pfile = pickle.load(open(args.pickle, 'rb'))
joblib.dump(pfile, args.joblib)
