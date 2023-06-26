from copy import copy
from random import shuffle
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple, Callable

import numpy
import numpy as np


class BatchCache(object):

    def __init__(self, shuffle: bool = False):
        self.shuffle = shuffle

    def add_batch(self, index: np.ndarray, features: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        pass

    def clear(self):
        pass

    def to_repeating_iterator(self, max_epochs: int = None, on_epoch_end: Callable[[int], None] = None):
        i = 0
        while i < max_epochs or max_epochs is None:
            for x in self:
                yield x

            if on_epoch_end is not None:
                on_epoch_end(i)

            i += 1


class MemCache(BatchCache):

    def __init__(self, shuffle: bool = False):
        super().__init__(shuffle)
        self.batches = []

    def add_batch(self, index: np.ndarray, features: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        self.batches.append((index, features, labels, weights))

    def __iter__(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        batches = copy(self.batches)
        if self.shuffle: shuffle(batches)
        return iter(batches)

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.batches[item]

    def __len__(self):
        return len(self.batches)


class FileCache(BatchCache):

    def __init__(self, shuffle: bool = False):
        super().__init__(shuffle)
        self.tmp_dir = TemporaryDirectory()
        self.batches = -1

        self._current_iter = -1

    def add_batch(self, index: np.ndarray, features: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        self.batches += 1
        numpy.savez(str(self.batches), index, features, labels, weights,)

    def __iter__(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        indexes = list(range(self.batches))
        if self.shuffle: shuffle(indexes)
        self._current_iter = iter(indexes)
        return self

    def __next__(self):
        return self[next(self._current_iter)]

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(f"{item}.npz") as data:
            return tuple([data[d] for d in data])

    def __len__(self):
        return self.batches

    def clear(self):
        self.tmp_dir.cleanup()
