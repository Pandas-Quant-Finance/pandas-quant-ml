from unittest import TestCase

import pandas as pd

from config import DF_TEST
from pandas_ml.transformers import LambertGaussianizer, LogNormalizer


class TestNormalizer(TestCase):

    def test_LogNormalizer(self):
        df_source = DF_TEST[["Close", "Volume"]].pct_change()

        pt = LogNormalizer()
        df = pt.transform(df_source)
        inv = pt.inverse(df)
        pd.testing.assert_frame_equal(inv, df_source.astype(float))

        pt = LogNormalizer(23)
        df = pt.transform(df_source)
        inv = pt.inverse(df)
        pd.testing.assert_frame_equal(inv, df_source.astype(float))

    def test_LambertGaussianizer(self):
        df_source = DF_TEST[["Close", "Volume"]].pct_change().dropna()
        pt = LambertGaussianizer()
        df = pt.transform(df_source)
        inv = pt.inverse(df)
        pd.testing.assert_frame_equal(inv, df_source.astype(float))
