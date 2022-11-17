import pandas as pd

from pandas_ta.pandas_ta_utils.decorators import for_each_top_level_row, for_each_top_level_column, \
    convert_series_as_data_frame
from pandas_ta_ml._abstract import Transformer
from .percent_change import PercentChange


@for_each_top_level_row
@for_each_top_level_column
@convert_series_as_data_frame
def ml_zscore(df: pd.DataFrame, period: int, ddof=1, exponential=False):
    return ZScore(period, ddof, exponential).transform(df)


@for_each_top_level_row
@for_each_top_level_column
@convert_series_as_data_frame
def ml_distance_from_ma(df: pd.DataFrame, period: int, exponential=False):
    return DistanceFormAverage(period, exponential).transform(df)


class ZScore(Transformer):

    def __init__(self, period, ddof=1, exponential=False):
        super().__init__()
        self.period = period
        self.ddof = ddof
        self.exponential = exponential
        self.percent_change = PercentChange()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.exponential:
            mean = (df.ewm(span=self.period) if self.period > 1 else df.ewm(alpha=self.period)).mean()
            std = (df.ewm(span=self.period) if self.period > 1 else df.ewm(alpha=self.period)).std(ddof=self.ddof)
        else:
            mean = df.rolling(self.period).mean()
            std = df.rolling(self.period).std(ddof=self.ddof)

        zscores = (df - mean) / std
        if zscores.ndim == 1:
            zscores.name = df.name
        else:
            zscores.columns = df.columns.tolist()

        return pd.concat(
            [self.percent_change.transform(mean.dropna()), std, zscores],
            keys=["mean", "std", "z"],
            axis=1
        ).swaplevel(-1, -2, axis=1)

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.swaplevel(0, -1, axis=1)
        ma = self.percent_change.inverse(df["mean"].dropna())
        df = df.loc[ma.index]
        return ma + df["z"].values * df["std"].values


class DistanceFormAverage(Transformer):

    def __init__(self, period, exponential=False):
        super().__init__()
        self.period = period
        self.exponential = exponential
        self.percent_change = PercentChange()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.exponential:
            mean = (df.ewm(span=self.period) if self.period > 1 else df.ewm(alpha=self.period)).mean()
        else:
            mean = df.rolling(self.period).mean()

        return pd.concat(
            [self.percent_change.transform(mean.dropna()), df / mean.values - 1],
            keys=["mean", "dist"],
            axis=1
        ).swaplevel(-1, -2, axis=1)

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.swaplevel(0, -1, axis=1)
        ma = self.percent_change.inverse(df["mean"].dropna())
        df = df.loc[ma.index]
        return ma * (1 + df["dist"].values)
