from __future__ import annotations
import numpy as np
import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class Winsorize(DataTransformer):

    def __init__(self, halflife_winsorise: int, vol_threshold: float):
        super().__init__()
        self.halflife_winsorise = halflife_winsorise
        self.vol_threshold = vol_threshold

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # winsorize using rolling 5X standard deviations to remove outliers
        ewm = df.ewm(halflife=self.halflife_winsorise)
        means = ewm.mean()
        stds = ewm.std()
        df = np.minimum(df, means + self.vol_threshold * stds)
        df = np.maximum(df, means - self.vol_threshold * stds)
        return df.dropna()
