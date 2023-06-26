from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_quant_ml.utils.batch_cache import FileCache


class TestFileCache(TestCase):

    def test_file_cache(self):
        cache = FileCache()
        batch = (pd.date_range('1990-01-01', periods=10), np.random.random(10), np.random.random(10), np.random.random(10))
        cache.add_batch(*batch)
        np.testing.assert_array_almost_equal(cache[0], batch)
        cache.clear()
