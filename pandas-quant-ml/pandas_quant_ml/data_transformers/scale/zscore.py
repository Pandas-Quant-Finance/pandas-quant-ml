from __future__ import annotations

from typing import Iterable, Callable, List

import pandas as pd

from pandas_df_commons.indexing.multiindex_utils import unique_level_values
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer
from sklearn.preprocessing import StandardScaler


class RollingZScore(DataTransformer):

    def __init__(self, period: int, demean: bool = True):
        super().__init__()
        self.period = period
        self.demean = demean

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.demean:
            df = (df - df.rolling(self.period).mean()) / df.rolling(self.period).std()
        else:
            df /= df.rolling(self.period).std()

        df = df.dropna()
        return df


class ZScaler(DataTransformer):

    def __init__(self, demean=True, with_std=True):
        super().__init__()
        self.scaler = StandardScaler(with_mean=demean, with_std=with_std)

    def _fit(self, df: pd.DataFrame):
        self.scaler = self.scaler.fit(df.values)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self.scaler.transform(df.values)
        return pd.DataFrame(transformed, index=df.index, columns=df.columns)
