from __future__ import annotations

import pandas as pd

from pandas_df_commons.indexing.datetime_indexing import extend_time_indexed_dataframe
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class PredictDateTimePeriods(DataTransformer):

    def __init__(self, periods, unit: str = None, only_weekdays: bool = True):
        super().__init__()
        self.periods = -abs(periods)

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.shift(self.periods).dropna()

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        # this is actually not really inverting the data by un-shifting. instead,
        # we need to extend the first level of the index such that we predict n periods
        extend_time_indexed_dataframe(df, abs(self.periods), timestep, only_weekdays)