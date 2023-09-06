from __future__ import annotations

import logging
from typing import Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from pandas_quant_ml.utils.batch_cache import BatchCache

_LOG = logging.getLogger(__name__)


class TorchDataGenerator(Dataset):

    def __init__(
            self,
            batch_cache: BatchCache,
            on_batch_end: Callable[[int, np.ndarray], None] = None,
            on_epoch_end: Callable[[int], None] = None,
            dtype: torch.Type | Tuple[torch.Type, ...] = torch.float32,
    ):
        super().__init__()
        self.batch_cache = batch_cache
        self._on_batch_end = on_batch_end
        self._on_epoch_end = on_epoch_end
        self.dtype = dtype
        self.epoch = 0

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch_cache[index]
        try:
            if not isinstance(self.dtype, tuple):
                return tuple(torch.from_numpy(b).type(self.dtype) for b in batch[1:])
            else:
                return tuple(torch.from_numpy(b).type(t) for b, t in zip(batch[1:], self.dtype))
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
