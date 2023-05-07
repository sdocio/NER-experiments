import argparse
import spacy
import sys
import warnings
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

    @classmethod
    def token_to_iob(cls, token):
        return (
            token.ent_iob_ if token.ent_iob_ == 'O'
            else f'{token.ent_iob_}-{token.ent_type_}'
        )


def parse_args():
    description = "Predict and evaluate NER tags using a spaCy model"

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
        default='model-best',
        type=str,
        metavar='MODEL',
        help='model'
    )

    return parser.parse_args()


args = parse_args()
iob = IOB()

nlp = spacy.load(args.model)
sentences = [
    [token[0] for token in sent] for sent in iob.parse_file(args.dataset)
]

predict = []
for tokens in tqdm(sentences):
    predict.append([
        (token.text, IOB.token_to_iob(token))
        for token in nlp(spacy.tokens.Doc(nlp.vocab, tokens))
    ])

for sent in predict:
    for token in sent:
        print(" ".join(token))
    print()
