from __future__ import annotations

from typing import Iterable, Callable

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class ZScore(DataTransformer):

    def __init__(self, period: int, names: str | Iterable[str] | Callable[[str, int], str] = None):
        super().__init__()
        self.period = period
        self.names = names

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df /= df.rolling(self.period).std()
        if self.names is None:
            return df
        else:
            return df.rename(columns=dict(zip(df.columns, self.names)) if isinstance(self.names, (list, tuple)) else self.names)
