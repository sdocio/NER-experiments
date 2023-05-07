"""
Code adapted from https://github.com/yuhaozhang/neural-ner
"""
import argparse
import os
import random
import sys
import torch
import warnings
from seqeval.metrics import classification_report as seq_classification_report
from tqdm import tqdm
from neuralner.trainer import Trainer
from neuralner.utils import torch_utils, constant
from neuralner.utils.vocab import Vocab
from neuralner.utils.dataloader import DataLoader

warnings.filterwarnings("ignore")


def parse_args():
    description = "Predict and evaluate NER tags using a LSTM torch model"

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
        default='model/best_model.pt',
        type=str,
        metavar='MODEL',
        help='model file',
    )
    parser.add_argument(
        '-e',
        '--eval',
        action='store_true',
        default=False,
        help='Evaluation only',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=torch.cuda.is_available(),
    )

    return parser.parse_args()


args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
random.seed(args.seed)
model_file = args.model
model_dir = os.path.dirname(model_file)
opt = torch_utils.load_config(model_file, device=device)

if args.gpu:
    opt['cpu'] = False
    opt['cuda'] = True
    torch.cuda.manual_seed(args.seed)
else:
    opt['cpu'] = True
    opt['cuda'] = False

trainer = Trainer(opt, cuda=args.gpu)
trainer.load(model_file, device=device)
vocab = Vocab(os.path.join(model_dir, 'vocab.pkl'))
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."  # noqa

char_vocab = Vocab(os.path.join(model_dir, 'vocab_char.pkl'))
assert opt['char_vocab_size'] == char_vocab.size, "Char vocab size must match that in the saved model."  # noqa

batch = DataLoader(args.dataset, opt, vocab, char_vocab)

label2id = constant.TYPE_TO_ID_IOB
id2label = dict([(v, k) for k, v in label2id.items()])

predictions = []
for i, b in enumerate(tqdm(batch)):
    preds, _ = trainer.predict(b)
    predictions += preds
predictions = [[id2label[p] for p in ps] for ps in predictions]

golden = batch.gold()
words = batch.words()

if not args.eval:
    assert len(golden) == len(words) == len(predictions), "Dataset size mismatch."  # noqa
    for ws, gs, ps in zip(words, golden, predictions):
        assert len(ws) == len(gs) == len(ps), "Example length mismatch."
        for w, g, p in zip(ws, gs, ps):
            print("{} {}".format(w, p))
        print()

results = seq_classification_report(golden, predictions, digits=3)
print(results, file=sys.stderr)
