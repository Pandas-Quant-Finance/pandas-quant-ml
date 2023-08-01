from __future__ import annotations

from functools import partial
from typing import Iterable, Callable, List, Tuple, Set

import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers._utils import renaming_columns
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
        self.names = names if callable(names) else renaming_columns(names)

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        daily_vol = df.pct_change().ewm(span=self.vola_lookback, min_periods=self.vola_lookback).std()

        return pd.concat(
            [((df.pct_change(periods=p) * self.target_vola_scale) / (daily_vol * np.sqrt(p)))\
                 .rename(columns=partial(self.names, i=i, p=p)) for i, p in enumerate(self.periods)],
            axis=1
        ).sort_index().dropna()


class NormaliseReturns(DataTransformer):

    # NOTE that you loose quite a lot of data if you use this class. It is better to use CalcNormalisedReturns
    #  because there we calculate the vola together with the returns and not afterward!

    def __init__(
            self,
            vola_lookback: int,
            target_vola_scale: float = 1.0,
    ):
        super().__init__()
        self.vola_lookback = vola_lookback
        self.target_vola_scale = target_vola_scale

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df.ewm(span=self.vola_lookback, min_periods=self.vola_lookback).std()
        return ((df * self.target_vola_scale) / vol).dropna()

