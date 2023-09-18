from unittest import TestCase

import pandas as pd

from pandas_df_commons.hashing import hash_df
from pandas_quant_ml.utils.obj_file_cache import ObjectFileCache
from testing_data.data import get_x_or


class TestObjCache(TestCase):

    def test_cache_df(self):
        df = get_x_or()

        cache = ObjectFileCache(
            lambda: df,
            "df",
            cache_key="a_cache_key",
            hash_func=lambda key: hash_df(df) * 31 + hash(key)
        )

        a, was_cached = cache.get_item("lala")
        self.assertFalse(was_cached)
        pd.testing.assert_frame_equal(a, df)

        b, was_cached = cache.get_item("lala")
        self.assertTrue(was_cached)
        pd.testing.assert_frame_equal(b, df)

        b, was_cached = cache.get_item("lala")
        self.assertTrue(was_cached)
        pd.testing.assert_frame_equal(b, df)
        pd.testing.assert_frame_equal(b, cache["lala"])
