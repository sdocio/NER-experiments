"""
The script splits an IOB2 file into batches of 10 sentences each, up
to N (--number) sentences in total.
"""
import argparse
import sys
import random

SENTENCES = 1500


def parse_args():
    """Parses script arguments"""
    description = ("Split an IOB2 file into batches of 10 sentences "
                   "each, up to N sentences in total.")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='IOB2 file',
    )
    parser.add_argument(
        '-n',
        '--number',
        type=int,
        default=SENTENCES,
        help='total number of sentences (total batches = N//10)',
    )

    return parser.parse_args()


def read_sentences_from_file(ifile):
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


args = parse_args()
sents = []

for sent in read_sentences_from_file(args.dataset):
    sents.append(sent)

assert len(sents) >= args.number, (
    "The value {args.number} must be <= the "
    f"number of sentences in the corpus ({len(sents)})"
)
random_sents = random.sample(sents, k=args.number)

num = 0
for num in range(args.number//10):
    print(f'random_batch_{num}.iob')
    with open(f'random_batch_{num}.iob', 'w') as ofile:
        ofile.write('\n'.join(random_sents[num * 10:(num*10)+10]) + '\n')

remain = args.number % 10
if remain != 0:
    filename = f"random_batch_{num + 1 if num != 0 else 0}.iob"
    print(filename)
    with open(filename, 'w') as ofile:
        ofile.write('\n'.join(random_sents[-remain:]) + '\n')
