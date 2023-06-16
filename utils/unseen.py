"""
This script splits an IOB corpus into two subsets: 'unseen' (sentences with
entities not seen during training) and 'seen' (sentences with at least one
entity seen during training).

The 'seen' subset may also include unseen entities if they coexist in a
sentence with an entity seen during training.

The script generates two output files: 'unseen_test.iob' and 'seen_test.iob'.
"""
import argparse
import sys
from collections import defaultdict


class IOBChunk:
    def __init__(self, token, iob, tag):
        self.tokens = [token]
        self.iob = [iob]
        self.tag = tag if iob == 'O' else tag[2:]
        self.chunks = []

    def get(self):
        return (' '.join(self.tokens), self.tag)

    def add(self, token, iob, tag):
        tag = tag if iob == 'O' else tag[2:]

        if self.tag != tag:
            raise ValueError(f"Wrong sequence: {self.iob[-1]}-{self.tag} {iob}-{tag}")  # noqa
        self.tokens.append(token)
        self.iob.append(iob)
        return self


def read_sentences_from_file(ifile):
    """Read IOB sentences from file, one at a time"""
    raw_sentence = ""
    try:
        with open(ifile) as fhi:
            for line in fhi:
                if line == "\n":
                    if raw_sentence == "":
                        continue
                    yield raw_sentence.rstrip()
                    raw_sentence = ""
                    continue

                if line:
                    raw_sentence += line

        # yield remaining contents if file does not end in '\n\n'
        if raw_sentence:
            yield raw_sentence.rstrip()
    except IOError:
        print("Unable to read file: " + ifile)
        sys.exit()


def get_dataset_from_file(ifile):
    return [
        [tuple(t.split(' ')) for t in sentence.split('\n')]
        for sentence in read_sentences_from_file(ifile)
    ]


def merge_entities(data, only_ents=False):
    chunks = []
    entities = defaultdict(set)

    subchunk = None
    for token in data:
        text, tag = token
        if tag.startswith('B-'):
            if subchunk is not None:
                chunks.append(subchunk.get())
                entities[subchunk.get()[1]].add(subchunk.get()[0])
            subchunk = IOBChunk(text, 'B', tag)
        elif tag.startswith('I-'):
            if subchunk is None:
                raise ValueError(f"Wrong sequence: O {tag}")
            subchunk.add(text, 'I', tag)
        elif tag == 'O':
            if subchunk is not None:
                chunks.append(subchunk.get())
                entities[subchunk.get()[1]].add(subchunk.get()[0])
                subchunk = None
            chunks.append(IOBChunk(text, 'O', tag).get())
        else:
            raise ValueError(f"Wrong tag: {tag}")
    if subchunk is not None:
        chunks.append(subchunk.get())
        entities[subchunk.get()[1]].add(subchunk.get()[0])

    if only_ents:
        return entities
    return chunks


def print_sentence(sentence, ofile):
    for token in sentence:
        print(" ".join(list(token)), file=ofile)
    print(file=ofile)


def parse_args() -> argparse.Namespace:
    """Parses script arguments"""
    description = ("Split a test IOB corpus into seen (entities seen at "
                   "training time) and unseen (entities not seen at training "
                   "time).")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'train',
        type=str,
        help='train dataset in IOB2 format',
    )
    parser.add_argument(
        'test',
        type=str,
        help='test dataset in IOB2 format',
    )

    return parser.parse_args()


args = parse_args()
train_sentences = get_dataset_from_file(args.train)
train_entities = defaultdict(set)
for sent in train_sentences:
    ents = merge_entities(sent, only_ents=True)
    for enttag in ents:
        for ent in ents[enttag]:
            train_entities[enttag].add(ent)

n_sents = 0
stats = defaultdict(int)
test_sentences = get_dataset_from_file(args.test)
try:
    unseen = open("unseen_test.iob", "w")
    seen = open("seen_test.iob", "w")
except IOError as exp:
    print("Operation failed: %s" % exp.strerror)

for sent in test_sentences:
    n_sents += 1
    flag_seen = 0
    flag_unseen = 0
    ents = merge_entities(sent, only_ents=True)
    if len(ents) == 0:
        if stats["empty_toks_seen"] > stats["empty_toks_unseen"]:
            print_sentence(sent, unseen)
            stats["nr_empty_sents_unseen"] += 1
            stats["empty_toks_unseen"] += len(sent)
        else:
            print_sentence(sent, seen)
            stats["nr_empty_sents_seen"] += 1
            stats["empty_toks_seen"] += len(sent)
        continue

    for enttag in ents:
        for ent in ents[enttag]:
            if ent in train_entities[enttag]:
                flag_seen += 1
            else:
                flag_unseen += 1

    # All entities present in the segment were not seen in the training corpus
    if flag_unseen > 0 and flag_seen == 0:
        print_sentence(sent, unseen)
        stats["nr_sents_unseen"] += 1
        stats["nr_ents_unseen"] += flag_unseen
    # Some entities present in the segment were seen in the training corpus
    else:
        print_sentence(sent, seen)
        stats["nr_sents_seen"] += 1
        stats["nr_ents_seen"] += (flag_seen + flag_unseen)

    stats["total_nr_ents_seen"] += flag_seen
    stats["total_nr_ents_unseen"] += flag_unseen

unseen.close()
seen.close()

print(f'Processed {n_sents} sentences.')
print(f'{stats["empty_toks_seen"]} empty tokens added to seen '
      f'({stats["nr_empty_sents_seen"]} sentences)')
print(f'{stats["empty_toks_unseen"]} empty tokens added to unseen '
      f'({stats["nr_empty_sents_unseen"]} sentences)')
print(f'{stats["total_nr_ents_seen"]} seen entities found, of which '
      f'{stats["nr_ents_seen"]} were put on the seen split')
print(f'{stats["total_nr_ents_unseen"]} unseen entities found, of which '
      f'{stats["nr_ents_unseen"]} were put on the unseen split')
