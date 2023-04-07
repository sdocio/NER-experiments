import argparse
import pickle
import sys
from features import CRFFeatures
from config import version
from IOB import IOB


def parse_args():
    description = ""

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    format = parser.add_mutually_exclusive_group(required=True)
    format.add_argument(
        '-t',
        '--text',
        action='store_true',
        help='Input file in text format',
    )
    format.add_argument(
        '-i',
        '--iob',
        action='store_true',
        help='Input file in IOB2 format',
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
        help='use POS tags as feature',
    )
    parser.add_argument(
        '-m',
        '--model',
        default='crf.model',
        type=str,
        metavar='FILE',
        help='model file',
    )
    parser.add_argument(
        '-s',
        '--spacy',
        type=str,
        required=False,
        metavar='FILE',
        help='spacy model for the tokenizer'
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
crf = pickle.load(open(args.model, 'rb'))
feats = CRFFeatures(with_pos=args.with_pos)

if args.text:
    textfile = ""
    with open(args.dataset) as ifile:
        for line in ifile:
            textfile += line.strip()

    import spacy
    try:
        spacy_model = args.spacy or spacy.util.get_installed_models()[0]
    except IndexError:
        print("Unable to load spaCy model", file=sys.stderr)
        sys.exit(1)

    nlp = spacy.load(spacy_model)
    doc = nlp(textfile)
    if args.with_pos:
        sentences = [[(t.text, t.pos_) for t in sent] for sent in doc.sents]
    else:
        sentences = [[(t.text,) for t in sent] for sent in doc.sents]
elif args.iob:
    sentences = [
        [(token[0],) for token in sent]
        for sent in iob.parse_file(args.dataset)
    ]

X = [feats.sent2features(s) for s in sentences]
y_pred = crf.predict(X)

for i, sentence in enumerate(sentences):
    for j, token in enumerate(sentence):
        print("{} {}".format(token[0], y_pred[i][j]))
    print()
