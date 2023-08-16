from __future__ import annotations

from typing import Any, Dict, Tuple, Iterable, Generator, Callable

import numpy as np
import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.utils.batch_cache import BatchCache


class Model(object):

    def __init__(self, looper: TrainTestLoop, **kwargs):
        self.looper = looper
        self.history: Dict[str, np.ndarray] | None = None

    def fit(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            nth_row_only: int = None,
            **kwargs,
    ) -> Callable[[], Generator[Tuple[Any, pd.DataFrame], None, None]]:
        self.history = self._fit(self.looper.train_test_iterator(frames, nth_row_only), **kwargs)

        def y_true_y_hat():
            for prediction in self.predict(frames, include_labels=True):
                yield prediction

        return y_true_y_hat

    def _fit(self, train_test_val: Tuple[BatchCache, ...], **kwargs) -> Dict[str, np.ndarray]:
        pass

    def predict(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            include_labels: bool = False,
    ) -> Generator[Tuple[Any, pd.DataFrame], None, None]:
        for prediction in self.looper.inference_generator(frames, self.get_model_predictor_from_numpy(), include_labels):
            yield prediction

    def get_model_predictor_from_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        pass