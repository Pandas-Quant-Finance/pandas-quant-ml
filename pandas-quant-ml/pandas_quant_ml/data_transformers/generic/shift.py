from __future__ import annotations

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class PredictPeriods(DataTransformer):

    def __init__(self, periods):
        super().__init__()
        self.periods = -abs(periods)

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.shift(self.periods).dropna()

