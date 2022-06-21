import argparse
import pickle
import spacy
import sys
from features import CRFFeatures
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
        '-s',
        '--spacy',
        choices=spacy.util.get_installed_models() or ["No models available"],
        type=str,
        required=True,
        metavar='FILE',
        help='spacy model for the tokenizer'
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        metavar='input file',
        type=str,
        help='dataset file'
    )
    return parser.parse_args()


args = parse_args()
nlp = spacy.load(args.spacy)
crf = pickle.load(open(args.model, 'rb'))
feats = CRFFeatures(with_pos=args.with_pos)

ifile = sys.stdin
if args.dataset is not None:
    ifile = open(args.dataset)

textfile = ""
for line in ifile:
    textfile += line.strip()

doc = nlp(textfile)
if args.with_pos:
    sentences = [[(t.text, t.pos_) for t in sent] for sent in doc.sents]
else:
    sentences = [[(t.text,) for t in sent] for sent in doc.sents]

X = [feats.sent2features(s) for s in sentences]
y_pred = crf.predict(X)

for i, sentence in enumerate(sentences):
    for j, token in enumerate(sentence):
        print("{} {}".format(token[0], y_pred[i][j]))
    print()
