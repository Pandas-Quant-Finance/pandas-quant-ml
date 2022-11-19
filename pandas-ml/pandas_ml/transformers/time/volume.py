from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from pandas_df_commons.indexing.decorators import foreach_top_level_row, foreach_column
from pandas_ml._abstract import Transformer


# all columns have the same index, so only rows levels are needed
@foreach_top_level_row
def ml_volume_time(df: pd.DataFrame, volume="Volume", kind='quadratic', fill_value='extrapolate'):
    return EvenlySpacedVolumeTime(volume=volume, kind=kind, fill_value=fill_value).transform(df)


class EvenlySpacedVolumeTime(Transformer):
    """
    This is a transformer on the x value (time) with interpolation of y values to volume time.
    However, the inverse only transforms volume time to human time and leaves the values unchanged !!
    """

    def __init__(self, volume="Volume", kind='quadratic', fill_value='extrapolate'):
        super().__init__()
        self.volume = volume
        self.kind = kind
        self.fill_value = fill_value
        self.affine_time_model = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df[self.volume].cumsum()
        idx = np.linspace(vol.iloc[0], vol.iloc[-1], len(df), endpoint=True)
        interpolator = partial(interp1d, vol, kind=self.kind, fill_value=self.fill_value)

        # fit a linear function to map from volume to unix timestamp
        time = df.index[0].timestamp(), df.index[-1].timestamp()
        self.affine_time_model = LinearRegression().fit(idx.reshape((-1, 1)), np.linspace(*time, len(idx)))

        @foreach_column
        def interpolate(s):
            return pd.Series(interpolator(s)(idx), index=idx, name=s.name)

        return interpolate(df)

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index(pd.to_datetime(self.affine_time_model.predict(df.index.values.reshape((-1, 1))), unit='s'))