from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from pandas_df_commons.indexing.datetime_indexing import extend_time_indexed_dataframe
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class PredictDateTimePeriods(DataTransformer):

    def __init__(self, periods, unit: timedelta = timedelta(days=1), only_weekdays: bool = True):
        super().__init__()
        self.periods = -abs(periods)
        self.unit = unit
        self.only_weekdays = only_weekdays

    def _fit(self, df: pd.DataFrame):
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.shift(self.periods).dropna()

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        # this is actually not really inverting the data by un-shifting. instead,
        # we need to extend the first level of the index such that we predict n periods
        if df.index.nlevels > 1 and not isinstance(df.index[0][0], pd.Timestamp):
            raise ValueError("Only Timestamp MultiLevelIndex are supported")

        # extend the dataframe by predicted time-steps and shift the data to the corresponding index
        periods = abs(self.periods)
        df, counts = extend_time_indexed_dataframe(df, periods, self.unit, self.only_weekdays, return_level_counts=True)

        return df.shift(int(periods * np.prod(counts)))
