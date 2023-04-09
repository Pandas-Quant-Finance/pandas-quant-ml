from __future__ import annotations

from typing import Generator
import random

import numpy as np
import pandas as pd

from pandas_df_commons._utils.streaming import window
from pandas_df_commons.indexing._utils import get_top_level_rows, top_level_separator_generator


def training_loop(
        df: pd.DataFrame,
        feature_columns: list | str,
        label_columns: list | str,
        batch_size: int = 128,
        feature_look_back_window: int = None,
        label_look_back_window: int = None,
        shuffle: bool = False,
) -> Generator[pd.DataFrame, None, None]:
    feature_generator = batch_generator(
        df[feature_columns], batch_size=batch_size, look_back_window=feature_look_back_window, shuffle=shuffle)

    for feature_index, feature_batch in feature_generator:
        if label_look_back_window is None:
            label_batch = df.loc[feature_index, label_columns].values
            yield feature_batch, label_batch
        else:
            # we would need to find df.loc[index-window:index] for each index
            raise NotImplemented


def batch_generator(
        df: pd.DataFrame,
        batch_size: int = 128,
        look_back_window: int = None,
        shuffle: bool = False,
) -> Generator[pd.DataFrame, None, None]:
    top_level_rows = get_top_level_rows(df)
    top_level_columns = None  # get_top_level_columns(df, level=col_level) if column_aggregator is not None else None
    index_generator = top_level_separator_generator(df, top_level_rows, top_level_columns, 0, shuffle_rows=shuffle)

    def sample():
        for (tl_col, tl_row), sub_df in index_generator:
            if look_back_window:
                windows = window(sub_df, look_back_window)
            else:
                def gen():
                    for i in range(0, len(sub_df), batch_size):
                        f = sub_df.iloc[i: i+batch_size]
                        if shuffle: f = f.sample(frac=1)
                        yield f
                windows = gen()

            if shuffle:
                windows = [w for w in windows]
                random.shuffle(windows)

            for w in windows:
                yield tl_row, w

    batch, indices = [], []
    for tlr, s in sample():
        if look_back_window:
            if len(batch) >= batch_size:
                yield pd.Index(indices), np.array(batch)
                batch, indices = [], []

            batch.append(s.values)
            indices.append(s.index[-1] if tlr is None else (tlr, s.index[-1]))
        else:
            yield s.index if tlr is None else pd.MultiIndex.from_product([[tlr], s.index]), s.values

    if len(batch) > 0: yield pd.Index(indices), np.array(batch)
