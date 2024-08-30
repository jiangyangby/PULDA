import itertools
import numpy as np
from torch.utils.data.sampler import Sampler

# Code obtained from:
# https://github.com/CuriousAI/mean-teacher/blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/mean_teacher/data.py
class MySampler(Sampler):
    def __init__(self, p_inds, u_inds, p_batch_size, u_batch_size):
        self.p_inds = p_inds
        self.u_inds = u_inds # self.u_inds = np.setdiff1d(u_inds, p_inds)

        self.p_batch_size = p_batch_size
        self.u_batch_size = u_batch_size
        
    def __iter__(self):
        p_iter = iterate_eternally(self.p_inds)
        u_iter = iterate_once(self.u_inds)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(p_iter, self.p_batch_size),
                    grouper(u_iter, self.u_batch_size))
        )

    def __len__(self):
        return len(self.u_inds) // self.u_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
