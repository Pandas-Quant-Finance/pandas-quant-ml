from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class Filter(DataTransformer):

    def __init__(
            self,
            filter: Callable[[pd.DataFrame], Iterable[bool]]
    ):
        super().__init__()
        self.filter = filter

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.filter(df)]

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        return df
