

from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_df_commons.extensions.functions import rolling_apply
from pandas_df_commons.indexing.decorators import foreach_top_level_row, convert_series_as_data_frame, foreach_column, \
    foreach_top_level_column
from pandas_ml._abstract.transfromer import Transformer


class ShiftAppend(Transformer):

    def __init__(self, period):
        super().__init__()
        self.period = period

    def _transform(self, df: pd.DataFrame):
        @foreach_column
        def shift(df):
            return pd.DataFrame({f"t-{i}": df.shift(i) for i in range(self.period, -1, -1)}, index=df.index)

        return shift(df)

    def _inverse(self, df: pd.DataFrame, base=None):
        @foreach_top_level_column
        def last(df):
            return df.iloc[:, -1]

        return last(df).droplevel(-1, axis=1)


class CumProd(Transformer):

    def __init__(self, period, base=1.0, offset=1.0):
        super().__init__()
        self.period = period
        self.offset = offset
        self.base = base

    def _transform(self, df: pd.DataFrame):
        self.base = df.iloc[0]

        @foreach_column
        def cumprod(df):
            return rolling_apply(df, self.period, lambda x: (x.iloc[:, 0] + self.offset).cumprod() - self.offset)

        return cumprod(df)

    def _inverse(self, df: pd.DataFrame, base=None):
        pct_change = df.iloc[:, 0]
        price = (pct_change + self.offset).cumprod().apply(lambda x: x * self.base, axis=1)
        price.join(df.iloc[:, 1:]).apply(lambda r: [r[0] * r[i] + self.offset for i in range(1, df.shape[1])], axis=1)

        return (df + 1).cumprod().apply(lambda x: x * self.base, axis=1)

