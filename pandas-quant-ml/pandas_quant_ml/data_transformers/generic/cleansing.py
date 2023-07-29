from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class CleanNaN(DataTransformer):

    def __init__(
            self,
            nan: Literal['None'] | float = 'drop',
            pinf: float = np.nan,
            ninf: float = np.nan,
    ):
        super().__init__()
        self.nan = nan
        self.pinf = pinf
        self.ninf = ninf

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace(np.inf, self.pinf).replace(-np.inf, self.ninf)
        return df.replace(np.nan, self.nan) if isinstance(self.nan, (float, int)) else df.dropna()

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        return df
