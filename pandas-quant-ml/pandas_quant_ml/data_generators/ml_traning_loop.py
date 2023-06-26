from __future__ import annotations

from typing import Tuple, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from pandas_df_commons._utils.streaming import Window
from pandas_df_commons.indexing._utils import get_top_level_rows, top_level_separator_generator


class TrainingLoop(object):

    def __init__(
            self,
            df: pd.DataFrame,
            feature_columns: list | str,
            label_columns: list | str,
            batch_size: int = 128,
            feature_look_back_window: int = None,
            label_look_back_window: int = None,
            shuffle: bool = False,
            label_weight_columns: list | str | None = None,
            label_extractor: Callable[[pd.DataFrame, np.ndarray, Any, Any], np.ndarray] | None = None,
            label_transformer: Callable[[np.ndarray], np.ndarray] | None = None,
            epochs: int = 1,
            progress: bool = False,
    ):
        super().__init__()
        self.df = df
        #self.feature_columns = feature_columns
        self.label_columns = label_columns
        #self.batch_size = batch_size
        #self.shuffle = shuffle
        #self.feature_look_back_window = feature_look_back_window
        self.label_look_back_window = label_look_back_window
        self.label_weight_columns = label_weight_columns
        self.label_extractor = label_extractor
        self.label_transformer = label_transformer
        self.epochs = epochs
        self.progress = progress

        self.feature_generator = BatchGenerator(
            self.df[feature_columns],
            batch_size=batch_size,
            look_back_window=feature_look_back_window,
            shuffle=shuffle
        )

        self._iterator = None

    def __getitem__(self, item):
        # we don't care about the epochs, we just create a bach and return the __getitem__
        return self._add_label(*self.feature_generator[item])

    def __iter__(self):
        def make_iterator():
            for epoch in tqdm(range(self.epochs), total=self.epochs) if self.progress else range(self.epochs):
                for feature_index, feature_batch in self.feature_generator:
                    yield self._add_label(feature_index, feature_batch)

        self._iterator = make_iterator()
        return self

    def __next__(self):
        if self._iterator is None:
            iter(self)

        return next(self._iterator)

    def _add_label(self, feature_index, feature_batch):
        if self.label_look_back_window is None:
            if self.label_extractor is None:
                label_batch = self.df.loc[feature_index, self.label_columns]
            else:
                label_batch = self.label_extractor(self.df, feature_batch, feature_index, self.label_columns)
        else:
            # we need to find df.(i)loc[index-window:index] for each index
            label_batch = []
            for i in feature_index:
                iloc = self.df.index.get_loc(i) + 1
                label_batch.append(self.df.iloc[iloc - self.label_look_back_window:iloc][self.label_columns].values)
            label_batch = np.array(label_batch)

        # convert to numpy
        if isinstance(label_batch, (pd.DataFrame, pd.Series)):
            label_batch = label_batch.values

        if self.label_transformer is not None:
            label_batch = self.label_transformer(label_batch)

        if self.label_weight_columns is not None:
            weight_batch = self.df.loc[feature_index, self.label_weight_columns].values
            return feature_batch, label_batch, weight_batch
        else:
            return feature_batch, label_batch

    def __len__(self):
        return len(self.feature_generator)


class BatchGenerator(object):

    def __init__(
            self,
            df: pd.DataFrame,
            batch_size: int = 128,
            look_back_window: int = None,
            shuffle: bool = False,
    ):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.look_back_window = look_back_window

        self.batched_windows = [
            Batch(Window(sub_df, look_back_window, shuffle=shuffle), batch_size or len(sub_df), top_level_row=tl_row)
            for (_, tl_row), sub_df in top_level_separator_generator(
                self.df,
                get_top_level_rows(self.df),
                None,
                shuffle_top_level_rows=shuffle
            )
        ]

        # we need to know the cum length of all batches
        self.batched_windows_length = []
        for b in self.batched_windows:
            self.batched_windows_length.append(
                len(b) + (
                    self.batched_windows_length[-1] if len(self.batched_windows_length) > 0 else 0
                )
            )

        self._iterator = None

    def __iter__(self):
        def make_iterator():
            for windows in self.batched_windows:
                for batch in windows:
                    yield self.__return__(batch)

        self._iterator = make_iterator()
        return self

    def __next__(self) -> Tuple[pd.Index, np.ndarray]:
        if self._iterator is None:
            iter(self)

        try:
            return next(self._iterator)
        except StopIteration as si:
            self._iterator = iter(self)
            raise si

    def __getitem__(self, item) -> Tuple[pd.Index, np.ndarray]:
        offset = 0
        for i, b in zip(self.batched_windows_length, self.batched_windows):
            if item < i:
                return self.__return__(b[item - offset])

            offset += i

        raise IndexError()

    def __len__(self):
        return self.batched_windows_length[-1]

    @staticmethod
    def __return__(value):
        if len(value) <= 0: return value

        # value[0] is a list of series or window frames
        # value[1] is top level row index
        series_or_frames, batch_kwargs = value

        isframe = series_or_frames[0].ndim > 1
        tlr = batch_kwargs["top_level_row"]

        # we want to return a pandas index and a numpy array
        if tlr is not None:
            index = pd.MultiIndex.from_tuples([(tlr, s.index[-1] if isframe else s.name) for s in series_or_frames])
        else:
            index = pd.Index([s.index[-1] if isframe else s.name for s in series_or_frames])

        # stack values along new axis 0 which is the batch dimension
        data = np.stack(sf.values for sf in series_or_frames)

        return index, data
