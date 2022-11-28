from unittest import TestCase

import pandas as pd
from config import DF_TEST
from pandas_quant_ml.transformers import ml_zscore, PercentChange, PositionalBar, ml_distance_from_ma, \
    RelativeBar, ZScore, DistanceFormAverage, GapUpperLowerBody


class TestStationizer(TestCase):

    def test_PositionalBar(self):
        df = DF_TEST[["Open", "High", "Low", "Close"]]
        pb = PositionalBar()
        pd.testing.assert_frame_equal(
            pb.inverse(pb.transform(df)),
            df,
            rtol=0.0001  #0.009  # FIXME one percent error is too much!
        )

    def test_PercentChange(self):
        pt = PercentChange()
        df = pt.transform(DF_TEST)
        inv = pt.inverse(df)
        pd.testing.assert_frame_equal(inv, DF_TEST.astype(float))

    def test_RelativeBar(self):
        pt = RelativeBar()
        df = pt.transform(DF_TEST)
        inv = pt.inverse(df)
        pd.testing.assert_frame_equal(inv, DF_TEST[["Open", "High", "Low", "Close", "Volume"]].astype(float))

    def test_GapUpperLowerBody(self):
        pt = GapUpperLowerBody()
        df = pt.transform(DF_TEST)
        inv = pt.inverse(df)
        pd.testing.assert_frame_equal(inv, DF_TEST[["Open", "High", "Low", "Close", "Volume"]].astype(float))

    #
    # single price
    #

    def test_ZScore(self):
        t = ZScore(60)
        df = t.transform(DF_TEST)
        inv = t.inverse(df)

        pd.testing.assert_frame_equal(inv[-2:], DF_TEST[-2:].astype(float))

    def test_ml_zscore(self):
        df = ml_zscore(DF_TEST["Close"], 60)
        dfe = ml_zscore(DF_TEST[["Close"]], 60, exponential=True)

        self.assertAlmostEqual(df["Close", "z"].min().item(), -5.421706575532131)
        self.assertAlmostEqual(df["Close", "z"].max().item(), 3.2811200265462)
        self.assertAlmostEqual(dfe["Close", "z"].min().item(), -4.113459697124262)
        self.assertAlmostEqual(dfe["Close", "z"].max().item(), 2.4606351566433577)

    def test_DistanceFormAverage(self):
        t = DistanceFormAverage(60)
        df = t.transform(DF_TEST)
        inv = t.inverse(df)

        pd.testing.assert_frame_equal(inv[-2:], DF_TEST[-2:].astype(float))

    def test_ml_distance_from_ma(self):
        df = ml_distance_from_ma(DF_TEST[["Close"]], 60)
        dfe = ml_distance_from_ma(DF_TEST[["Close"]], 60, exponential=True)

        self.assertAlmostEqual(df["Close", "dist"].min().item(), -0.28136357697796)
        self.assertAlmostEqual(df["Close", "dist"].max().item(), 0.14829821277935795)
        self.assertAlmostEqual(dfe["Close", "dist"].min().item(), -0.2615217631715002)
        self.assertAlmostEqual(dfe["Close", "dist"].max().item(), 0.10071129555069813)
