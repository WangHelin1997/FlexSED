import torch
import numpy as np
import yaml
import os
from torch.utils.data import Sampler


def load_yaml_with_includes(yaml_file):
    def loader_with_include(loader, node):
        # Load the included file
        include_path = os.path.join(os.path.dirname(yaml_file), loader.construct_scalar(node))
        with open(include_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    yaml.add_constructor('!include', loader_with_include, Loader=yaml.FullLoader)

    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def customized_lr_scheduler(optimizer, warmup_steps=10000, decay_steps=1e6, end_factor=1e-4):
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=min(1 / warmup_steps, 1),
                                end_factor=1.0, total_iters=warmup_steps)

    decay_scheduler = LinearLR(optimizer,
                               start_factor=1.0,
                               end_factor=end_factor,
                               total_iters=decay_steps)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], 
                             milestones=[warmup_steps])
    return scheduler


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


class ConcatDatasetBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, epoch=0):
        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):
        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]
            min_len = min(c_len, min_len)
        return min_len