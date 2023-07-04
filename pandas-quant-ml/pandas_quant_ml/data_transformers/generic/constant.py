from __future__ import annotations

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class DataConstant(DataTransformer):

    def __init__(self, constant, name):
        super().__init__()
        self.constant = constant
        self.name = name

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame({}, index=df.index)
        df[self.name] = self.constant
        return df
