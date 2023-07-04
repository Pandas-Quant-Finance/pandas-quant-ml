import numpy as np
import pytest

from pandas_quant_ml.utils.batch_cache import MemCache, FileCache


@pytest.mark.parametrize(
    "cache",
    [
        MemCache,
        FileCache
    ]
)
class TestBaching:

    def test_iterator(self, cache):
        cache = cache()
        cache.add_batch(np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10))
        cache.add_batch(np.ones(10), np.ones(10), np.ones(10), np.ones(10))

        assert len(list(cache)) == 2, f"{type(cache)} failed to iterate"

    def test_get_item(self, cache):
        cache = cache()
        cache.add_batch(np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10))
        cache.add_batch(np.ones(10), np.ones(10), np.ones(10), np.ones(10))

        assert cache[0][0].sum() == 0, f"{type(cache)} failed to get item"
        assert cache[1][0].sum() == 10, f"{type(cache)} failed to get item"

    def test_stacking(self, cache):
        cache = cache()
        cache.add_batch(np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10))
        cache.add_batch(np.ones(10), np.ones(10), np.ones(10), np.ones(10))

        new_cache = cache.concatenate(cache)
        assert len(new_cache) == 1
        assert len(new_cache[0][0]) == 20

