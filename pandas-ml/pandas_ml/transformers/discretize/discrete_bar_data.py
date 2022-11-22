import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import KBinsDiscretizer

from pandas_df_commons.indexing.decorators import foreach_top_level_row, foreach_top_level_column
import pandas as pd

from pandas_ml._abstract import Transformer


@foreach_top_level_row
@foreach_top_level_column
def ml_categorical_bar(df: pd.DataFrame, open="Open", high="High", low="Low", close="Close", body_threshold=0.975, gap_threshold=0.002):
    return CategoricalBar(
        open=open, high=high, low=low, close=close, body_threshold=body_threshold, gap_threshold=gap_threshold
    ).transform(df)


@foreach_top_level_row
@foreach_top_level_column
def ml_kbins_discretizer(df: pd.DataFrame, discretizer: KBinsDiscretizer):
    return KBins(discretizer).transform(df)


class CategoricalBar(Transformer):
    """
    We try to classify single candle sticks based on simple rules to integers [-5, 5]:
        t = threshold i.e. 97.5%

        red, green candles:
          g: positive candle
          r: negative candle

        opening gaps
          positive opening gap (x3, x1)
          negative opening gap (x1, x3)
          no gap (x2 / x2)

        shapes:
            https://blog.quantinsti.com/candlestick-patterns-meaning/
            1. body 99 % of hl
            2. lower shadow > upper shadow && shadows > body
            3. body > upper + lower shadow
            4. shadows > body
            5. upper shadow > lower shadow  && shadows > body

    :param df: data frame containing the open, high, low, close data
    :return: integer number of category: 2 * 3 * 5 = 30
    """

    def __init__(self, open="Open", high="High", low="Low", close="Close", body_threshold=0.975, gap_threshold=0.002):
        super().__init__()
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.body_threshold = body_threshold
        self.gap_threshold = gap_threshold

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["close-1"] = df[self.close].shift(1)

        def map(df):
            o, h, l, c, c1 = df[self.open], df[self.high], df[self.low], df[self.close], df["close-1"]
            hl = (h / l - 1)
            gap = o / c1 - 1
            sign, body_upper, body_lower, rank = (1, c, o, [5, 4, 1, 2, 3]) if c > o else (-1, o, c, [-5, -1, -4, -2, -3])
            body = body_upper / body_lower - 1
            lower_shadow = body_lower / l - 1
            upper_shadow = h / body_upper - 1
            shadow = lower_shadow + upper_shadow

            if abs(gap) > self.gap_threshold:
                if sign > 0:
                    gap = 3 if gap > 0 else 1
                else:
                    gap = 3 if gap < 0 else 1
            else:
                gap = 2

            if body >= (hl * self.body_threshold):
                return rank[0] * gap
            else:
                if shadow > body:
                    if lower_shadow > (upper_shadow + body):
                        return rank[1] * gap
                    elif upper_shadow > (lower_shadow + body):
                        return rank[2] * gap
                    else:
                        return rank[3] * gap
                else:
                    return rank[4] * gap

        ranks = df.apply(map, raw=False, axis=1)
        return ranks.rename("bar_category").to_frame()

    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df["bar_category"].cumsum()


class KBins(Transformer):

    def __init__(self, discretizer: KBinsDiscretizer):
        super().__init__()
        self.discretizer = discretizer
        self.columns = None

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            discrete = self.discretizer.transform(df)
        except NotFittedError:
            discrete = self.discretizer.fit_transform(df)

        self.columns = df.columns.tolist()
        return pd.DataFrame(
            discrete if isinstance(discrete, np.ndarray) else discrete.toarray(),
            index=df.index
        )

    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.discretizer.inverse_transform(df),
            index=df.index,
            columns=self.columns
        )