from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.math import col_div
from pandas_quant_ml.data_transformers.filter.outlier import Winsorize
from pandas_quant_ml.data_transformers.generic.lambda_transform import Lambda
from pandas_quant_ml.data_transformers.generic.selection import Select, SelectJoin
from pandas_quant_ml.data_transformers.generic.shift import PredictDateTimePeriods
from pandas_quant_ml.data_transformers.generic.windowing import MovingWindow
from pandas_quant_ml.data_transformers.normalizer.normalized_returns import CalcNormalisedReturns
from pandas_quant_ml.data_transformers.scale.zscore import RollingZScore
from pandas_quant_ml.data_transformers.stationizer.returns import Returns
from pandas_ta.technical_analysis import ta_macd
from testing_data import DF_AAPL


class TestGenericTransformer(TestCase):

    def test_select(self):
        dt = Select("Close", names=["Lala"])

        df, queue = dt.fit_transform(DF_AAPL, 20, True)
        inv = dt.inverse(df, queue)
        pd.testing.assert_frame_equal(DF_AAPL[["Close"]], inv)
        self.assertFalse(dt.transform(DF_AAPL, False)[1])

    def test_select_join(self):
        pass

    def test_shift(self):
        pass

    def test_window(self):
        pass

    def test_constant(self):
        pass

    def test_lambda(self):
        pass
