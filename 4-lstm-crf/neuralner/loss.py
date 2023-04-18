import torch
from torch import nn
from neuralner.utils import constant


def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constant.PAD_TYPE_ID] = 0
    crit = nn.CrossEntropyLoss(weight)
    return crit
