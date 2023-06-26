from __future__ import annotations

import pandas as pd

from pandas_df_commons._utils.streaming import Window
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class MovingWindow(DataTransformer):

    def __init__(self, periods: int, step: int = 1, axis=0):
        super().__init__()
        self.period = periods
        self.step = step
        self.axis = axis

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        w = Window(df, self.period)
        indexes = [i for i in range(0, len(w), self.step)]

        if indexes[-1] < len(w) - 1:
            indexes.append(len(w) - 1)

        return pd.concat([w[i] for i in indexes], keys=range(len(indexes)), axis=self.axis)

