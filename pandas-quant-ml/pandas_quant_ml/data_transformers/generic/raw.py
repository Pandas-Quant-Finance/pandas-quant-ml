from __future__ import annotations

import pandas as pd

from pandas_df_commons._utils.streaming import Window
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class ToNumpy(DataTransformer):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

        self._index = None
        self._columns = None

    def _fit(self, df: pd.DataFrame):
        self._index = df.index.tolist()
        self._columns = df.columns.tolist()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = df.values

        if self.shape:
            raw = raw.reshape(self.shape)

        return raw

    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if self.shape:
            df = df.reshape((-1, len(self._columns)))

        return pd.DataFrame(df, index=self._index, columns=self._columns)
