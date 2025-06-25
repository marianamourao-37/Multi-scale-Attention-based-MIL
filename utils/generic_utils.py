import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
import gc 

import nltk
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device(args):
    return "cuda" if args.device == "cuda" else 'cpu'


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        #torch.backends.cudnn.benchmark = False

def get_Paths(args):
    chk_pt_path = Path(f"{args.checkpoints}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")
    output_path = Path(f"{args.output_path}/{args.dataset}/zz/{args.model_type}/{args.arch}/{args.root}")
    tb_logs_path = Path(f"{args.tensorboard_path}/{args.dataset}/{args.model_type}/{args.arch}/{args.root}")

    return chk_pt_path, output_path, tb_logs_path


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def clear_memory():
    gc.collect()  # Run Python's garbage collector to clear unused CPU memory
    torch.cuda.empty_cache()  # Clear unused GPU memory cache
    
def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n

        if param.requires_grad:
            num_params_train += n

    # Format total parameters
    if num_params >= 1e6:
        total_params_str = f"{num_params / 1e6:.6f} M"
    elif num_params >= 1e3:
        total_params_str = f"{num_params / 1e3:.3f} K"
    else:
        total_params_str = f"{num_params}"

    # Format trainable parameters
    if num_params_train >= 1e6:
        total_trainable_params_str = f"{num_params_train / 1e6:.6f} M"
    elif num_params_train >= 1e3:
        total_trainable_params_str = f"{num_params_train / 1e3:.3f} K"
    else:
        total_trainable_params_str = f"{num_params_train}"

    print(f"\nTotal number of parameters: {total_params_str}")
    print(f"Total number of trainable parameters: {total_trainable_params_str}")


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='w'):

    file = h5py.File(output_path, mode)
    
    for key, val in asset_dict.items():
        data_shape = val.shape
        data_type = val.dtype
        chunk_shape = (1, ) + data_shape[1:]
        maxshape = (None, ) + data_shape[1:]

        dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
        dset[:] = val

        if attr_dict is not None:
            if key in attr_dict.keys():
                for attr_key, attr_val in attr_dict[key].items():
                    
                    dset.attrs[attr_key] = attr_val
        file.close()
    return output_path
