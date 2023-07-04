from __future__ import annotations

import pandas as pd

from pandas_df_commons._utils.streaming import Window
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class MovingWindow(DataTransformer):

    def __init__(self, periods: int, axis=0):
        super().__init__()
        self.period = periods
        self.axis = axis

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        w = Window(df, self.period)

        all_windows = list(w)
        return pd.concat(
            all_windows,
            keys=[(w.index if self.axis==0 else w.columns)[-1] for w in all_windows],
            axis=self.axis
        ).sort_index()

