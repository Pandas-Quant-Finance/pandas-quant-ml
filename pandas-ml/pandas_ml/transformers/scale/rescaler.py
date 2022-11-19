from typing import Tuple

import pandas as pd

from pandas_df_commons.indexing.decorators import foreach_top_level_row, foreach_top_level_column
from pandas_ml._abstract import Transformer
from pandas_ml.utils.rescaler import ReScaler


@foreach_top_level_column
@foreach_top_level_row
def ml_rescale(df: pd.DataFrame, range=(-1, 1), domain=None, digits=None, axis=None):
    r = Rescale(range, domain, )
    if axis is not None:
        return df.apply(lambda x: ml_rescale(x, range, domain, digits, None), raw=False, axis=axis, result_type='broadcast')
    else:
        if domain is None:
            domain = (df.values.min(), df.values.max())

        rescaled = ReScaler(domain, range)(df.values)

        if digits is not None:
            rescaled = np.around(rescaled, digits)

        if rescaled.ndim > 1:
            return pd.DataFrame(rescaled, columns=df.columns, index=df.index)
        else:
            return pd.Series(rescaled, name=df.name, index=df.index)


class Rescale(Transformer):

    def __init__(self, range: Tuple[int, int] = (-1, 1), domain: Tuple[int, int] = None, digits=None, axis=0):
        super().__init__()
        self.axis = axis
        self.range = range
        self.digits = digits
        self._domain = None
        if domain is None:
            self.domain = lambda df: (df.values.min(), df.values.max())
        else:
            self.domain = lambda df: domain

    def transform(self, df: pd.DataFrame):
        self._domain = self.domain(df)
        rescaler = ReScaler(self._domain, self.range)
        if self.digits is not None:
            return np.around(df.apply(rescaler, axis=self.axis), self.digits)
        else:
            return df.apply(rescaler, axis=self.axis)

    def inverse(self, df: pd.DataFrame):
        assert self._domain is not None, "Need to apply transform first before inverse is possible"
        rescaler = ReScaler(self.range, self._domain)
        if self.digits is not None:
            return np.around(df.apply(rescaler, axis=self.axis), self.digits)
        else:
            return df.apply(rescaler, axis=self.axis)
