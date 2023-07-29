from unittest import TestCase

import pandas as pd

from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.data_transformers.generic.shift import PredictDateTimePeriods
from pandas_quant_ml.data_transformers.normalizer.normalized_returns import CalcNormalisedReturns, NormaliseReturns
from pandas_quant_ml.data_transformers.stationizer.returns import Returns
from tesing_data import DF_AAPL


class TestNormalizerTransformer(TestCase):

    def test_normalize_returns(self):
        pipeline1 = Select("Close") \
            >> CalcNormalisedReturns(1, 60, 0.15, names="target_return_1")

        pipeline2 = Select("Close") \
            >> Returns(1, names="target_return_1") \
            >> NormaliseReturns(60, 0.15)

        pd.testing.assert_frame_equal(
            pipeline1.fit_transform(DF_AAPL, 0)[0].tail(5),
            pipeline2.fit_transform(DF_AAPL, 0)[0].tail(5),
        )
