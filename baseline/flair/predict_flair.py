import argparse
import logging.config
import sys
import warnings
from flair.nn import Classifier
from flair.data import Sentence
from tqdm import tqdm

warnings.filterwarnings("ignore")


class IOB:
    def __init__(self, sep=" "):
        self._sep = sep

    def parse_file(self, ifile):
        return [
            self._parse_sentence(raw)
            for raw in self._read_sentences_from_file(ifile)
        ]

    def _parse_sentence(self, raw_sentence):
        return [
            tuple(token.split(self._sep))
            for token in raw_sentence.strip().split("\n")
        ]

    def _read_sentences_from_file(self, ifile):
        raw_sentence = ""
        try:
            with open(ifile) as fhi:
                for line in fhi:
                    if line == "\n":
                        if raw_sentence == "":
                            continue
                        yield raw_sentence
                        raw_sentence = ""
                        continue

                    if line:
                        raw_sentence += line

            if raw_sentence:
                yield raw_sentence
        except IOError as err:
            print(err, file=sys.stderr)
            sys.exit()


def parse_args():
    description = "Predict and evaluate NER tags using a Flair model"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='IOB input file',
    )
    parser.add_argument(
        '-m',
        '--model',
        default='flair/ner-spanish-large',
        type=str,
        metavar='MODEL',
        help='model'
    )
    return parser.parse_args()


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

args = parse_args()
iob = IOB()
ner = Classifier.load(args.model)

sentences = [
    [token[0] for token in sent] for sent in iob.parse_file(args.dataset)
]

predict = []
for tokens in tqdm(sentences):
    tagged = [[t, "O"] for t in tokens]
    sentence = Sentence(tokens)
    ner.predict(sentence)

    for e in sentence.get_spans('ner'):
        for i, tok in enumerate(e.tokens):
            tagged[tok.idx - 1][1] = f'B-{e.tag}' if i == 0 else f'I-{e.tag}'
    predict.append(tagged)

for sent in predict:
    for token in sent:
        print(" ".join(token))
    print()
