import os
import random
import time

import torch
import numpy as np
from tensorboardX import SummaryWriter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for cpu
    torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))


def numpy_fix_init(worker_id):
    np.random.seed(2 << 16 + worker_id)


numpy_init_dict = {"train": numpy_random_init, "val": numpy_fix_init, "test": numpy_fix_init}


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    pre_path = path
    counter = 0
    while os.path.exists(path):     # force create folder
        path = f'{pre_path}{counter}'
        counter += 1
    os.makedirs(path)
    return path

def set_save_path(save_path, remove=True):
    forced_path = ensure_path(save_path, remove=remove)
    set_log_path(forced_path)
    writer = SummaryWriter(os.path.join(forced_path, 'tensorboard'))
    return log, writer, forced_path


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot
