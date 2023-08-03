from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_df_commons.indexing import get_columns
from pandas_df_commons.indexing.decorators import foreach_top_level_column
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class GapBodyUpperLower(DataTransformer):

    def __init__(self, open="Open", high="High", low="Low", close="Close", volume="Volume", drop_nan_volume=True):
        super().__init__()
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.drop_nan_volume = drop_nan_volume
        self.basis = {}

    def _fit(self, df: pd.DataFrame):
        # FIXME use correct values for inverse functions
        self.basis["volume"] = get_columns(df, self.volume).iloc[0]
        self.basis["open"] = get_columns(df, self.open).iloc[0]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        @foreach_top_level_column
        def trans(df):
            o = df[self.open]
            c = df[self.close]
            h = df[self.high]
            l = df[self.low]
            oc = pd.concat([o, c], axis=1)
            c_1 = c.shift(1)
            gap = (o / c_1 - 1).fillna(0)

            # calculate close, upper_shadow, lower_shadow, body
            res = pd.DataFrame({
                "gap": gap,
                "body": (c / o - 1),
                "upper": (h / oc.max(axis=1) - 1),
                "lower": (oc.min(axis=1) / l - 1),
            }, index=df.index)

            if self.volume is not None and self.volume in df.columns:
                rel_vol = df[self.volume].pct_change().fillna(0)
                if not self.drop_nan_volume or not rel_vol.isnull().all():
                    res[self.volume] = rel_vol

            return res
        return trans(df)

    def _inverse(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        inv = pd.DataFrame(index=df.index)
        previous_close = self.basis["open"]
        open, close = np.empty((2, len(df)))

        for i in range(len(df)):
            open[i] = (1 + df["gap"].iloc[i]) * previous_close
            close[i] = open[i] * (1 + df["body"].iloc[i])
            previous_close = close[i]

        inv[self.open] = open
        inv[self.close] = close

        high = inv.max(axis=1) * (1 + df["upper"])
        low = inv.min(axis=1) / (1 + df["lower"])
        inv.insert(1, self.low, low)
        inv.insert(1, self.high, high)

        if self.volume in df.columns:
            inv[self.volume] = (df[self.volume] + 1).cumprod().apply(lambda x: x * self.basis["volume"])

        return inv

