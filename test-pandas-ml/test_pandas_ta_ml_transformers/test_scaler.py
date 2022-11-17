from unittest import TestCase

import pandas as pd

from config import DF_TEST
from pandas_ta_ml.transformers.scale import ml_rescale, Rescale


class TestScaler(TestCase):

    def test_rescale(self):
        df0 = ml_rescale(DF_TEST, axis=0)
        df01 = ml_rescale(DF_TEST, domain=(0, 1000), axis=0)
        df1 = ml_rescale(DF_TEST, axis=1)

        self.assertListEqual(df0.min(axis=0).tolist(), [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        self.assertListEqual(df1.min(axis=1).tolist(), [-1.0] * len(df1))
        self.assertAlmostEqual(df01["Volume"].min(), 9.4)

    def test_Rescale(self):
        r = Rescale()

        df = r.transform(DF_TEST)
        inv = r.inverse(df)
        pd.testing.assert_frame_equal(inv, DF_TEST.astype(float))

