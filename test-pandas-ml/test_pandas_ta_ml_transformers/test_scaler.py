from unittest import TestCase

import pandas as pd

from config import DF_TEST
from pandas_ml.transformers.scale import ml_rescale, Rescale
from pandas_ml.transformers.scale.accumulate import CumProd, ShiftAppend


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

    def test_ShiftAppend(self):
        t = ShiftAppend(3)
        df = DF_TEST[["Close", "Open"]]
        c = t.transform(df)
        i = t.inverse(c)

        self.assertListEqual(
            c.columns.tolist(),
            [
                ('Close', 't-3'), ('Close', 't-2'), ('Close', 't-1'), ('Close', 't-0'),
                ('Open', 't-3'), ('Open', 't-2'), ('Open', 't-1'), ('Open', 't-0'),
            ]
        )
        pd.testing.assert_frame_equal(i, df)

    def test_CumProd(self):
        t = CumProd(4)
        df = DF_TEST[["Close"]].pct_change()
        c = t.transform(df)
        i = t.inverse(c)

        pd.testing.assert_frame_equal(i, c)
