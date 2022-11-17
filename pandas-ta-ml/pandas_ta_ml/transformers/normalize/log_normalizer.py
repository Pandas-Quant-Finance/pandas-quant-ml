import numpy as np
import pandas as pd

from pandas_ta.pandas_ta_utils.decorators import for_each_column
from pandas_ta_ml._abstract.transfromer import Transformer


def ml_log(df: pd.DataFrame, base=None):
    return LogNormalizer(base).transform(df)


def ml_exp(df: pd.DataFrame, base=np.e):
    return LogNormalizer(base).inverse(df)


class LogNormalizer(Transformer):

    def __init__(self, base=None):
        super().__init__()
        self.base = base

    def transform(self, df: pd.DataFrame):
        @for_each_column  # because we need to check for illegal values (<=0) for each column individually
        def do_log(df):
            if df.min() <= 0:
                df = df + 1

            return np.log(df) / (1.0 if self.base is None else np.log(self.base))

        return do_log(df)

    def inverse(self, df: pd.DataFrame):
        base = (np.e if self.base is None else self.base)
        return (base ** df) - 1
