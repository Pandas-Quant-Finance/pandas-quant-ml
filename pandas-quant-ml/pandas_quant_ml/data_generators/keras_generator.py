from __future__ import annotations

import logging
from typing import Tuple, Callable

import numpy as np
from tensorflow import keras

from pandas_quant_ml.utils.batch_cache import BatchCache

_LOG = logging.getLogger(__name__)


class KerasDataGenerator(keras.utils.Sequence):

    # Custom Data Generator with keras.utils.Sequence
    # https://dzlab.github.io/dltips/en/keras/data-generator/

    def __init__(
            self,
            batch_cache: BatchCache,
            on_batch_end: Callable[[int, np.ndarray], None] = None,
            on_epoch_end: Callable[[int], None] = None,
    ):
        super().__init__()
        self.batch_cache = batch_cache
        self._on_batch_end = on_batch_end
        self._on_epoch_end = on_epoch_end
        self.epoch = 0

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = self.batch_cache[index]
        try:
            return batch[1:]
        finally:
            if self._on_batch_end is not None:
                self._on_batch_end(self.epoch, *batch)

    def __len__(self):
        return len(self.batch_cache)

    def on_end(self):
        self.batch_cache.clear()

    def on_epoch_end(self):
        if self._on_epoch_end is not None:
            self._on_epoch_end(self.epoch)

        self.epoch += 1
