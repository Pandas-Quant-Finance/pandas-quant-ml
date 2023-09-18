import tempfile
from pathlib import Path
from typing import Callable, Any, Tuple

from pandas_quant_ml.utils.serialize import serialize, deserialize
import os

CACHE_ROOT_DIR = Path(os.environ.get('PQML_CACHE_ROOT_DIR', tempfile.gettempdir()))


class ObjectFileCache(object):

    # partial(self._feature_pipelines[name].fit_transform, reserved_data_length=test_length, reset=reset_pipeline)
    def __init__(self, data_provider: Callable[[], Any], *path, cache_size: int = 1, hash_func: Callable[[Any], int] = hash):
        super().__init__()
        self.provider = data_provider
        self.path = CACHE_ROOT_DIR.joinpath(*path)
        self.cache_size = cache_size
        self.hash_func = hash_func

    def __getitem__(self, item):
        return self.get_item(item)[0]

    def get_item(self, arg, ) -> Tuple[Any, bool]:
        if self.cache_size == 0: return self.provider()

        obj_hash = str(self.hash_func(arg))
        if self.cache_size > 0: self._evict_old_but(obj_hash)

        cache_file = self.path.joinpath(obj_hash)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            cache_file.touch(exist_ok=True)
            return deserialize(cache_file), True

        res = self.provider()
        serialize(res, cache_file)
        return res, False

    def _evict_old_but(self, except_hash: str):
        cache_files = [f for f in self.path.glob("*") if f != except_hash]
        cache_files = list(sorted(cache_files, key=lambda f: f.lstat().st_mtime, reverse=True))
        for evictable_file in cache_files[self.cache_size:]:
            try:
                evictable_file.unlink(missing_ok=True)
            except Exception:
                pass
