from unittest import TestCase

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from config import DF_TEST
from pandas_ta_ml.transformers import CategoricalBar, KBins


class TestDiscretizer(TestCase):

    def test_CategoricalBar(self):
        t = CategoricalBar()
        df = t.transform(DF_TEST)
        idf = t.inverse(df)

        self.assertEquals(np.argmax(idf), np.argmax(DF_TEST["Close"]))

    def test_KBins(self):
        t = KBins(KBinsDiscretizer(111, strategy='uniform', encode='ordinal'))
        t2 = KBins(KBinsDiscretizer(111, strategy='uniform', encode='onehot'))

        f = DF_TEST[["Close"]].pct_change().fillna(0)
        df = t.transform(f)
        df2 = t2.transform(f)

        pd.testing.assert_frame_equal(t.inverse(df), t2.inverse(df2))
        pd.testing.assert_frame_equal(t.inverse(df), f, atol=0.002)

