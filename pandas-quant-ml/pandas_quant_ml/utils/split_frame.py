from typing import Tuple

import pandas as pd

from pandas_df_commons.indexing.multiindex_utils import unique_level_values


def split_frames(*frames: pd.DataFrame, test_length=None, split_index=None) -> Tuple[Tuple[pd.DataFrame, ...], Tuple[pd.DataFrame, ...]]:
    idx = unique_level_values(frames[0])
    no_split = test_length is None and split_index is None
    not_enough_data = test_length is not None and len(idx) - test_length <= 0

    if no_split or not_enough_data:
        return frames, tuple(f.iloc[:0] for f in frames)
    else:
        train_test_split_index = split_index if split_index is not None else idx[len(idx) - test_length]
        return tuple(f.loc[:train_test_split_index] for f in frames), tuple(f.loc[train_test_split_index:] for f in frames)


def get_split_index(df, test_length):
    split_idx = unique_level_values(df)
    idx = len(split_idx) - test_length

    if idx < 0 or idx >= len(split_idx):
        return None
    else:
        return split_idx[len(split_idx) - test_length]
