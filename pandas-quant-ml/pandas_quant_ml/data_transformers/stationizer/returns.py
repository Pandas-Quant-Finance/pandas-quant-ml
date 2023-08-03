from __future__ import annotations

from functools import partial
from string import Template
from typing import Iterable, Callable

import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers._utils import renaming_columns
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class Returns(DataTransformer):

    def __init__(self, periods: int|Iterable[int], names: str|Iterable[str]|Callable[[str, int], str]|Template=Template("${x}_returns_${p}")):
        super().__init__()
        self.periods = periods if isinstance(periods, Iterable) else [periods]
        self.names = names if callable(names) else renaming_columns(names)

        self._min_period_columns = None

    def _fit(self, df: pd.DataFrame):
        self._min_period_columns = \
            [df.rename(columns=partial(self.names, i=i, p=p), level=-1) for i, p in enumerate(self.periods)][0].columns

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [df.pct_change(periods=p).rename(columns=partial(self.names, i=i, p=p), level=-1) for i, p in enumerate(self.periods)],
            axis=1
        ).sort_index().dropna()

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        i = np.argmin(self.periods)
        p = self.periods[i]

        # find columns with min period
        df = df[self._min_period_columns]
        factors = df.join(prev_df[[]], how='outer').shift(-p) + 1
        factors.columns = prev_df.columns

        return (prev_df * factors).shift(p)
