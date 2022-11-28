import numpy as np

from pandas_df_commons.indexing.decorators import foreach_top_level_row, foreach_top_level_column
import pandas as pd

from pandas_quant_ml._abstract import Transformer


@foreach_top_level_row
@foreach_top_level_column
def ml_positional_bar(df, open="Open", high="High", low="Low", close="Close"):
    gap = df[close] / df[close].shift(1) - 1
    body = df[close] / df[open] - 1
    shadow = df[high] / df[low] - 1
    position = (df[high] + df[low] / 2) / (df[open] + df[close] / 2) - 1

    return pd.DataFrame(
        dict(gap=gap, body=body, shadow=shadow, position=position),
        index=df.index
    )


@foreach_top_level_row
@foreach_top_level_column
def ml_realative_bar(df: pd.DataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume", drop_nan_volume=True):
    rb = RelativeBar(open=open, high=high, low=low, close=close, volume=volume, drop_nan_volume=drop_nan_volume)
    return rb.transform(df)


@foreach_top_level_row
@foreach_top_level_column
def ml_gulb_bar(df: pd.DataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume", drop_nan_volume=True, **kwargs):
    rb = GapUpperLowerBody(open=open, high=high, low=low, close=close, volume=volume, drop_nan_volume=drop_nan_volume)
    return rb.transform(df)


class PositionalBar(Transformer):

    def __init__(self, open="Open", high="High", low="Low", close="Close"):
        super().__init__()
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.basis = 0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.basis = df[self.close].iloc[0]
        gap = (df[self.close] / df[self.close].shift(1) - 1).fillna(0)
        body = df[self.close] / df[self.open] - 1
        center = (df[self.open] + df[self.close]) / 2
        shadow = (df[self.high] / center) - 1 + (center / df[self.low]) - 1
        position = ((df[self.high] + df[self.low]) / 2) / center - 1

        return pd.DataFrame(
            dict(gap=gap, body=body, shadow=shadow, position=position),
            index=df.index
        )

    def _inverse(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        open, high, low, close = np.zeros((4, len(df)))
        previous_close = self.basis
        for i in range(len(df)):
            close[i] = previous_close * (1 + df["gap"].iloc[i])
            previous_close = close[i]

        open = close / (1 + df["body"].values)
        center = ((open + close) / 2) * (1 + df["position"].values)
        half_shadow = (df["shadow"].values / 2)
        high = center * (1 + half_shadow)
        low = center * (1 - half_shadow)

        return pd.DataFrame(
            dict(Open=open, High=high, Low=low, Close=close),
            index=df.index
        )


class RelativeBar(Transformer):

    def __init__(self, open="Open", high="High", low="Low", close="Close", volume="Volume", drop_nan_volume=True):
        super().__init__()
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.drop_nan_volume = drop_nan_volume
        self.basis = 0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.basis = df.iloc[0].copy()
        relative = pd.DataFrame(index=df.index)
        close_1 = df[self.close].shift(1)

        for col in [self.open, self.high, self.low, self.close]:
            relative[col] = (df[col] / close_1 - 1).fillna(0)

        if self.volume is not None:
            rel_vol = df[self.volume].pct_change().fillna(0)
            if not self.drop_nan_volume or not rel_vol.isnull().all():
                relative[self.volume] = rel_vol

        return relative

    def _inverse(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        inv = pd.DataFrame(index=df.index)

        # start with the restore of the close
        inv[self.close] = (df[self.close] + 1).cumprod().apply(lambda x: x * self.basis[self.close])

        # restore all other price columns
        for col in [self.open, self.high, self.low]:
            inv[col] = inv[[self.close]].shift(1).join(df[col] + 1)\
                            .apply(lambda x: x[self.close] * x[col], axis=1)\
                            .fillna(self.basis[col])

        # fix the ordering
        inv = inv.reindex(df.columns.tolist(), axis=1)

        # finally append the volume column if present
        if self.volume in df.columns:
            inv[self.volume] = (df[self.volume] + 1).cumprod().apply(lambda x: x * self.basis[self.volume])

        return inv


class GapUpperLowerBody(Transformer):

    def __init__(self, open="Open", high="High", low="Low", close="Close", volume="Volume", drop_nan_volume=True):
        super().__init__()
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.drop_nan_volume = drop_nan_volume
        self.basis = {}

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df[self.open]
        c = df[self.close]
        h = df[self.high]
        l = df[self.low]
        oc = pd.concat([o, c], axis=1)
        c_1 = c.shift(1)
        gap = (o / c_1 - 1).fillna(0)

        # calculate close, upper_shadow, lower_shadow, body
        res = pd.DataFrame({
            "gap": gap,
            "upper": (h / oc.max(axis=1) - 1),
            "lower": (oc.min(axis=1) / l - 1),
            "body": (c / o - 1)
        }, index=df.index)

        if self.volume is not None and self.volume in df.columns:
            rel_vol = df[self.volume].pct_change().fillna(0)
            if not self.drop_nan_volume or not rel_vol.isnull().all():
                res[self.volume] = rel_vol
                self.basis["volume"] = df[self.volume].iloc[0]

        self.basis["open"] = df[self.open].iloc[0]

        return res

    def _inverse(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        inv = pd.DataFrame(index=df.index)
        previous_close = self.basis["open"]
        open, close = np.empty((2, len(df)))

        for i in range(len(df)):
            open[i] = (1 + df["gap"].iloc[i]) * previous_close
            close[i] = open[i] * (1 + df["body"].iloc[i])
            previous_close = close[i]

        inv[self.open] = open
        inv[self.close] = close

        high = inv.max(axis=1) * (1 + df["upper"])
        low = inv.min(axis=1) / (1 + df["lower"])
        inv.insert(1, self.low, low)
        inv.insert(1, self.high, high)

        if self.volume in df.columns:
            inv[self.volume] = (df[self.volume] + 1).cumprod().apply(lambda x: x * self.basis["volume"])

        return inv

