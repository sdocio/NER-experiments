import sys
import random

SENTENCES = 1500


class IOB:
    def __init__(self, ifile):
        self._ifile = ifile

    def read_sentences_from_file(self):
        raw_sentence = ""
        try:
            with open(self._ifile) as fhi:
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


iob = IOB(sys.argv[1])
sentences = iob.read_sentences_from_file()
sents = []

for sent in sentences:
    sents.append(sent)

random_sents = random.choices(sents, k=SENTENCES)

for num in range(SENTENCES//10):
    print(f'random_batch_{num}.iob')
    with open(f'random_batch_{num}.iob', 'w') as ofile:
        ofile.write('\n'.join(random_sents[num * 10:(num*10)+10]) + '\n')
