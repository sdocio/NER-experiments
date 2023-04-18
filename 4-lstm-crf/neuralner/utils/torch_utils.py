import sys
import torch


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def load_config(filename, device=torch.device('cpu')):
    try:
        dump = torch.load(filename, map_location=device)
    except BaseException as ex:
        print("Model loading failed: {}".format(ex))
        sys.exit(1)
    return dump['config']
