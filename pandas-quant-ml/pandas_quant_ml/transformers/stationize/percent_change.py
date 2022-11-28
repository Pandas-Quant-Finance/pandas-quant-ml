from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_df_commons.indexing.decorators import foreach_top_level_row
from pandas_quant_ml._abstract.transfromer import Transformer


@foreach_top_level_row
def ml_pct_change(df: pd.DataFrame):
    return PercentChange().transform(df)


@foreach_top_level_row
def ml_cumprod(df: pd.DataFrame, base: float | pd.Series | np.ndarray = None):
    return PercentChange(base).transform(df)


class PercentChange(Transformer):

    def __init__(self, base=1.0):
        super().__init__()
        self.base = base

    def _transform(self, df: pd.DataFrame):
        self.base = df.iloc[0]
        return df.pct_change().fillna(0)

    def _inverse(self, df: pd.DataFrame, base=None):
        base = base if base else self.base
        return (df + 1).cumprod().apply(lambda x: x * base, axis=1)
