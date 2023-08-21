from __future__ import annotations

from typing import Callable, Iterable, Any, List

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class SwapLevels(DataTransformer):

    def __init__(
            self,
            level1: int,
            level2: int,
            axis=1,
    ):
        super().__init__()
        self.level1 = level1
        self.level2 = level2
        self.axis = axis

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.swaplevel(self.level1, self.level2, self.axis).sort_index(axis=self.axis)

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        return df.swaplevel(self.level2, self.level1, self.axis).sort_index(axis=self.axis)
