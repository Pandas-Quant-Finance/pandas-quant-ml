from __future__ import annotations

from functools import partial
from typing import Iterable, Callable

import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class CalcNormalisedReturns(DataTransformer):

    def __init__(
            self,
            periods: int|Iterable[int],
            vola_lookback: int,
            target_vola_scale: float = 1.0,
            names: str | Iterable[str] | Callable[[str, int], str] = None
    ):
        super().__init__()
        self.periods = periods if isinstance(periods, Iterable) else [periods]
        self.vola_lookback = vola_lookback
        self.target_vola_scale = target_vola_scale
        self.names = \
            names if callable(names) else lambda x, i: f"{x}_{names[i] if isinstance(names, Iterable) else names}"

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        daily_returns = df.pct_change()
        daily_vol = daily_returns.ewm(span=self.vola_lookback, min_periods=self.vola_lookback).std()

        return pd.concat(
            [((df.pct_change(periods=p) * self.target_vola_scale) / (daily_vol * np.sqrt(p)))\
                 .rename(columns=partial(self.names, i=i)) for i, p in enumerate(self.periods)],
            axis=1
        ).sort_index().dropna()

