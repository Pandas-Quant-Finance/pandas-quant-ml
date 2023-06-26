from __future__ import annotations

import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class VolatilityTargetReturn(DataTransformer):

    def __init__(self, period: int, vola_target: float, scale_factor: float = 252):
        super().__init__()
        self.period = period
        self.vola_target = vola_target
        self.scale_factor = np.sqrt(scale_factor)

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        vola = df.ewm(span=self.period, min_periods=self.period).std()
        df * self.vola_target / (vola * self.scale_factor).shift(1)
        return df
