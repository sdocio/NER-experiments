import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from neuralner import model, loss, crf
from neuralner.utils import constant, torch_utils, dataloader

def unpack_batch(batch, cuda):
    fsize = dataloader.INPUT_SIZE
    if cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:fsize]]
        labels = batch[fsize].cuda()
    else:
        inputs = [b if b is not None else None for b in batch[:fsize]]
        labels = batch[fsize]
    masks = inputs[1]
    orig_idx = batch[-1]
    return inputs, labels, masks, orig_idx

class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, opt, emb_matrix=None, joint=False, cuda=False):
        self.opt = opt
        self.cuda = cuda
        self.model = model.BLSTM_CRF(opt, emb_matrix)
        if opt['crf']:
            self.crit = crf.CRFLoss(opt['num_class'], True)
        else:
            self.crit = loss.SequenceLoss(opt['num_class'])
        self.parameters = [p for m in (self.model, self.crit) for p in m.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.crit.cuda()

    def predict(self, batch, unsort=True):
        inputs, labels, masks, orig_idx = unpack_batch(batch, self.cuda)
        self.model.eval()
        batch_size = inputs[0].size(0)
        logits = self.model(inputs)
        lens = list(masks.data.eq(0).long().sum(1).squeeze())
        if self.opt['crf']:
            loss, trans = self.crit(logits, masks, labels)
            predictions = []
            trans = trans.data.cpu().numpy()
            scores = logits.data.cpu().numpy()
            for i in range(batch_size):
                tags, _ = crf.viterbi_decode(scores[i,:lens[i]], trans)
                predictions += [tags]
        else:
            logits_flat = logits.view(-1, logits.size(-1))
            loss = self.crit(logits_flat, labels.view(-1))
            predictions = np.argmax(logits_flat.data.cpu().numpy(), axis=1)\
                .reshape([batch_size, -1]).tolist()
            predictions = [p[:l] for l,p in zip(lens, predictions)] # remove paddings
        if unsort:
            _, predictions = [list(t) for t in zip(*sorted(zip(orig_idx, predictions)))]
        return predictions, loss.data.item()

    def load(self, filename, device=torch.device('cpu')):
        try:
            checkpoint = torch.load(filename, map_location=device)
        except BaseException as ex:
            print("Cannot load model from {}: {}".format(filename, ex))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        if 'crit' in checkpoint:
            self.crit.load_state_dict(checkpoint['crit'])
        self.opt = checkpoint['config']


