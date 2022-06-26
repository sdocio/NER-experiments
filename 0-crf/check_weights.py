import argparse
import eli5
import pickle
import sys
from config import version


def parse_args():
    description = ""

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
        '-m',
        '--model',
        default='crf.model',
        type=str,
        metavar='FILE',
        help='model file'
    )
    return parser.parse_args()


args = parse_args()
try:
    crf = pickle.load(open(args.model, 'rb'))
except FileNotFoundError:
    print("Error: file {} was not found".format(args.model))
    sys.exit(1)
except Exception:
    print("Error: unable to load the model in {}".format(args.model))
    sys.exit(127)

explain = eli5.explain_weights(crf)
print(eli5.format_as_text(explain))
