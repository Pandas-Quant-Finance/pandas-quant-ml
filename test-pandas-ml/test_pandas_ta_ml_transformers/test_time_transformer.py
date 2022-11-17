from unittest import TestCase

import numpy as np

from config import DF_TEST
from pandas_ta_ml.transformers.time.volume import EvenlySpacedVolumeTime


class TestTimeTransformer(TestCase):

    def test_EvenlySpacedVolumeTime(self):
        t = EvenlySpacedVolumeTime()
        df = t.transform(DF_TEST)
        idf = t.inverse(df)

        np.testing.assert_almost_equal(df.values, idf.values)

        self.assertAlmostEqual((idf.index[0] - DF_TEST.index[0]).total_seconds(), 0, 5)
        self.assertAlmostEqual((idf.index[-1] - DF_TEST.index[-1]).total_seconds(), 0, 5)
