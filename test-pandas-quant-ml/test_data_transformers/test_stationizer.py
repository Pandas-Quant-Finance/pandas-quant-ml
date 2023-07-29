from unittest import TestCase

import pandas as pd

from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.data_transformers.generic.shift import PredictDateTimePeriods
from pandas_quant_ml.data_transformers.stationizer.returns import Returns
from tesing_data import DF_AAPL


class TestStationizerTransformer(TestCase):

    def test_returns(self):
        dt = Select("Close", names=['lala']) \
             >> Returns(1)

        df, q = dt.fit_transform(DF_AAPL, 20, True)
        inv = dt.inverse(df, q)
        pd.testing.assert_frame_equal(
            DF_AAPL[["Close"]].iloc[1:],
            inv.iloc[1:],
            rtol=1e-6
        )

    def test_returns2(self):
        dt = Select("Open", "Close", names=['o', 'c']) \
             >> Returns(range(1, 3))

        df, q = dt.fit_transform(DF_AAPL, 20, True)
        inv = dt.inverse(df, q)
        pd.testing.assert_frame_equal(
            DF_AAPL[["Open", "Close"]].iloc[2:],
            inv.iloc[2:],
            rtol=1e-6
        )
