import torch
from torch.utils.data import Sampler
import random

class BalancedFrameSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.event_indices = []
        self.background_indices = []

        for idx, (_, _, label) in enumerate(dataset.samples):
            if label == 0:
                self.background_indices.append(idx)
            else:
                self.event_indices.append(idx)

        self.num_batches = min(len(self.event_indices), len(self.background_indices)) // (batch_size // 2)

    def __iter__(self):
        random.shuffle(self.event_indices)
        random.shuffle(self.background_indices)

        for i in range(self.num_batches):
            start = i * (self.batch_size // 2)
            end = start + (self.batch_size // 2)

            batch_event = self.event_indices[start:end]
            batch_bg = self.background_indices[start:end]

            yield batch_event + batch_bg

    def __len__(self):
        return self.num_batches
