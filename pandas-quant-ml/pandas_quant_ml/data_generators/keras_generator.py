from __future__ import annotations
from threading import Lock
from typing import Callable, Any

from tensorflow import keras
import numpy as np
import pandas as pd

from pandas_df_commons.indexing._utils import get_top_level_rows
from .ml_traning_loop import TrainingLoop

import logging
_LOG = logging.getLogger(__name__)


class KerasDfDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            df: pd.DataFrame,
            feature_columns: str | list,
            label_columns: str | list,
            batch_size: int = 32,
            feature_look_back_window: int = None,
            label_look_back_window: int = None,
            shuffle: bool = False,
            label_weight_columns: list | str | None = None,
            label_extractor: Callable[[pd.DataFrame, np.ndarray, Any, Any], np.ndarray] | None = None,
            label_transformer: Callable[[np.ndarray], np.ndarray] | None = None,
            on_get_batch: Callable[[tuple], None]=None,
            sanitize: bool = True,
    ):
        super().__init__()
        self.on_get_batch = on_get_batch
        self.sanitize = sanitize

        self.training_loop = TrainingLoop(
            df,
            feature_columns, label_columns,
            batch_size,
            feature_look_back_window, label_look_back_window,
            shuffle,
            label_weight_columns, label_extractor, label_transformer
        )

    def label_and_prediction(self, model):
        y, y_hat = [], []
        for Xyw in self.training_loop:
            y.append(Xyw[1])
            y_hat.append(model.predict(Xyw[0]))

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        return y, y_hat

    def __getitem__(self, index):
        Xyw = self.training_loop[index]

        if self.sanitize:
            # TODO remove all .replace([np.inf, -np.inf], np.nan).dropna()
            #  x = x[numpy.isfinite(x)]
            Xyw = tuple(np.nan_to_num(i, copy=True, nan=0.0, posinf=0.0, neginf=0.0) for i in Xyw)

        if self.on_get_batch is not None:
            self.on_get_batch(index, *Xyw)

        return Xyw

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.training_loop)
