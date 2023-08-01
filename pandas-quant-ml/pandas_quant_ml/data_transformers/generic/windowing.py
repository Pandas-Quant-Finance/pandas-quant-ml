from __future__ import annotations

import pandas as pd

from pandas_df_commons._utils.streaming import Window
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class MovingWindow(DataTransformer):

    def __init__(self, periods: int, axis=0, multi_index=True):
        super().__init__()
        self.period = periods
        self.axis = axis
        self.multi_index = multi_index

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        w = Window(df, self.period)

        all_windows = list(w)
        if self.axis < 1:
            res = pd.concat(
                all_windows,
                keys=[w.index[-1] for w in all_windows],
                axis=self.axis
            ).sort_index()

            if not self.multi_index:
                res.index = res.index.tolist()

            return res
        else:
            columns = pd.MultiIndex.from_product([range(self.period), df.columns])
            return pd.concat(
                [pd.DataFrame(
                    w.values.reshape(1, -1),
                    index=[w.index[-1]],
                    columns=columns if self.multi_index else [f"{c[1]}_{c[0]}" for c in columns],
                ) for w in all_windows],
                axis=0
            )
