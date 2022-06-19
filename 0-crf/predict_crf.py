import argparse
import pickle
from IOB import IOB


__version = '0.0.2'


def parse_args():
    description = ""

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
        help='dataset file'
    )
    return parser.parse_args()


args = parse_args()
iob = IOB()

sentences = iob.parse_file(args.dataset)
crf = pickle.load(open(args.model, 'rb'))

if crf.with_pos and len(sentences[0][0]) < 2:
    raise("With PoS option, the text must include PoS tags.")

X = [crf.sent2features(s) for s in sentences]
y_pred = crf.predict(X)

for i, sentence in enumerate(sentences):
    for j, token in enumerate(sentence):
        print("{} {}".format(token[0], y_pred[i][j]))
    print()
