from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_df_commons.indexing.decorators import for_each_top_level_row
from pandas_ta_ml._abstract.transfromer import Transformer


@for_each_top_level_row
def ml_pct_change(df: pd.DataFrame):
    return PercentChange().transform(df)


@for_each_top_level_row
def ml_cumprod(df: pd.DataFrame, base: float | pd.Series | np.ndarray = None):
    return PercentChange(base).transform(df)


class PercentChange(Transformer):

    def __init__(self, base=1.0):
        super().__init__()
        self.base = base

    def transform(self, df: pd.DataFrame):
        self.base = df.iloc[0]
        return df.pct_change().fillna(0)

    def inverse(self, df: pd.DataFrame):
        return (df + 1).cumprod().apply(lambda x: x * self.base, axis=1)
