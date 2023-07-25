"""
Send prompts to GPT3 ("text-davinci-003"), trying to deal with batch and
request limits. It was used with few-shot learning for NER, providing two
examples and a tokenized text (see file "sample-prompt.txt"). The testing
dataset must be in IOB2 format.

Usage:

(1) One batch per request

TOKENIZERS_PARALLELISM=false OPENAI_API_KEY='your-secret-api-key' \
python3 test_gpt3.py -i sample-prompt.txt -b MAX_TOKENS test.iob

Process sentences in batches with size < MAX_TOKENS (default: 1500).

(2) One sentence per request

TOKENIZERS_PARALLELISM=false OPENAI_API_KEY='your-secret-api-key' \
python3 test_gpt3.py -i sample-prompt.txt --sentence test.iob

Process one sentence per request.
"""
import argparse
import logging
import os
import sys
import time
import warnings
import openai
from transformers import GPT2TokenizerFast, logging as hf_logging


SAFE_MARGIN = 250
LIMIT_PER_REQUEST = 4000
LIMIT_TOKENS_PER_MINUTE = 150000
LIMIT_REQUESTS_PER_MINUTE = 18
BATCH_LIMIT = 1500

# prompt with a few examples of the task being tested
INSTRUCTION = """Etiqueta un texto usando etiquetas NER y en formato IOB2. Por ejemplo, para el texto "La Universidad de Santiago tiene su sede principal en Santiago de Compostela.", la salida correcta es:

La O
Universidad B-ORG
de I-ORG
Santiago I-ORG
tiene O
su O
sede O
principal O
en O
Santiago B-LOC
de I-LOC
Compostela I-LOC
. O

Por ejemplo, para el texto "La Compostela es el certificado que Juan recibió tras hacer el Camino.", la salida correcta es:

La O
Compostela B-MISC
es O
el O
certificado O
que O
Juan B-PER
recibió O
tras O
hacer O
el O
Camino B-LOC
. O

El texto que debes etiquetar es"""  # noqa

hf_logging.set_verbosity(hf_logging.FATAL)
warnings.filterwarnings('ignore')


def read_sentences_from_file(ifile):
    """
    Read an IOB2 file and returns a generator with the sentences (one token
    per line).
    """
    raw_sentence = ""
    try:
        with open(ifile, encoding="utf8") as fhi:
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


def get_maxtokens(length):
    return LIMIT_PER_REQUEST - length - SAFE_MARGIN


def send_request(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    length = len(tokenizer(prompt)['input_ids'])
    max_tokens = get_maxtokens(length)

    response = openai.Completion.create(
        model="text-davinci-003",
        temperature=0,
        prompt=prompt,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    logger.info('  ** max_tokens_requested=%s', max_tokens)
    logger.info('  ** model={response["model"]}, object={response["object"]}, finish_reason={response["choices"][0]["finish_reason"]}')  # noqa
    logger.info('  ** tokens: prompt={response["usage"]["prompt_tokens"]}, result={response["usage"]["completion_tokens"]}, total={response["usage"]["total_tokens"]}')  # noqa

    process_response(response["choices"][0]["text"])

    return response["usage"]["total_tokens"]


def process_response(text):
    print(text.strip())
    if args.sentence:
        print()


def restricted_int(value):
    try:
        int_val = int(value)
        if isinstance(int_val, int) and 0 < int_val <= BATCH_LIMIT:
            return int_val
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f'{value} must be an integer between 1 and {BATCH_LIMIT}')


def get_instruction(value):
    instruct = ""
    try:
        with open(value, encoding="utf8") as ifile:
            instruct = "\n".join([line.strip() for line in ifile])
        return instruct
    except IOError as err:
        print(err, file=sys.stderr)
    raise argparse.ArgumentTypeError(
        f'Unable to read instructions from file {value}'
    )


def parse_args():
    description = "Testing GPT3"

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        'dataset',
        type=str,
        help='dataset to be tested',
    )
    parser.add_argument(
        "-d",
        "--debug",
        action='store_true',
        help='Debug log mode'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Verbose mode'
    )
    parser.add_argument(
        "-i",
        "--instruction",
        type=get_instruction,
        help='File containing the prompt with instructions for GPT3'
    )
    group.add_argument(
        "--batch",
        type=restricted_int,
        default=BATCH_LIMIT,
        metavar="MAX_TOKENS",
        help="Number of tokens per batch."
    )
    group.add_argument(
        "--sentence",
        action='store_true',
        default=False,
        help="Process one sentence per request instead of MAX_TOKENS batches"
    )

    return parser.parse_args()


args = parse_args()

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
screen = logging.StreamHandler()
screen.setFormatter(formatter)
screen.setLevel(logging.ERROR)
if args.debug:
    logger.setLevel(logging.DEBUG)
    screen.setLevel(logging.DEBUG)
elif args.verbose:
    logger.setLevel(logging.INFO)
    screen.setLevel(logging.INFO)
logger.addHandler(screen)
instruction = args.instruction or INSTRUCTION
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

sent_id = 0

current_toks = 0
batch = ''
batch_nr = 1
sentences = read_sentences_from_file(args.dataset)

start_time = time.time()
sum_tokens = 0
sum_requests = 1

for sent in sentences:
    sent_id += 1
    tokens = [tokens.split(" ")[0] for tokens in sent.split('\n')]
    current_toks += len(tokenizer(sent)['input_ids'])
    length_prompt = len(tokenizer(instruction)['input_ids']) + current_toks

    current_time = time.time()

    if args.sentence:
        batch = '\n'.join(tokens) + '\n'

        if(sum_tokens > LIMIT_TOKENS_PER_MINUTE
                or sum_requests > LIMIT_REQUESTS_PER_MINUTE):
            sleep = 60 - (current_time - start_time) + 15
            time.sleep(sleep)
            start_time = current_time = time.time()
            sum_tokens = 0
            sum_requests = 0
            logger.info(
                "Sleeping %s in request number %s", sleep, sum_requests)

        logger.info('Starting sentence number %s', sent_id)
        logger.info('  ** time consumed %s', (current_time - start_time))
        logger.info('  ** request number %s', sum_requests)
        logger.info('  ** sum of tokens %s', sum_tokens)
        sum_tokens += send_request(instruction + '\n\n' + batch)
        sum_requests += 1
    else:
        batch += '\n'.join(tokens) + '\n'

        if length_prompt > BATCH_LIMIT:
            logger.info('Starting batch number %s', batch_nr)
            logger.info('  ** last_sentence_id: %s', sent_id)
            logger.info('  ** time consumed %s', (current_time - start_time))
            logger.info('  ** request number %s', sum_requests)
            logger.info('  ** sum of tokens %s', sum_tokens)
            sum_tokens += send_request(instruction + '\n\n' + batch)
            sum_requests += 1
            batch_nr += 1
            current_toks = 0
            batch = ''

if args.sentence is False:
    sum_tokens += send_request(instruction + '\n\n' + batch)
