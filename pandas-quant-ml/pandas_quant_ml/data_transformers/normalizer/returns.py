from __future__ import annotations

from functools import partial
from typing import Iterable, Callable

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class Returns(DataTransformer):

    def __init__(self, periods: int|Iterable[int], names: str|Iterable[str]|Callable[[str, int], str]=None):
        super().__init__()
        self.periods = periods if isinstance(periods, Iterable) else [periods]
        self.names = \
            names if callable(names) else lambda x, i: f"{x}_{names[i] if isinstance(names, Iterable) else names}"

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [df.pct_change(periods=p).rename(columns=partial(self.names, i=i)) for i, p in enumerate(self.periods)],
            axis=1
        ).sort_index().dropna()

