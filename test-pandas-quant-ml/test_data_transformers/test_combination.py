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


class TestTransformerCombination(TestCase):

    def test_predict_returns(self):
        dt = Select("Open", "Close", names=['o', 'c']) \
             >> Returns(range(1, 3)) \
             >> PredictDateTimePeriods(1)

        df, q = dt.fit_transform(DF_AAPL, 20, True)
        inv = dt.inverse(df, q)
        print(inv)

