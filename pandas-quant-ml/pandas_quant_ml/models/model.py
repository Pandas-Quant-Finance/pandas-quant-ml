from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Iterable, Generator, Callable

import numpy as np
import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.utils.batch_cache import BatchCache

LOG = logging.getLogger(__name__)

class Model(object):

    def __init__(self, looper: TrainTestLoop, **kwargs):
        self.looper = looper
        self.history: Dict[str, np.ndarray] | None = None

    def fit(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            train_test_split_ratio: float | Tuple[float, float] = 0.75,
            *,
            batch_size: int = None,
            nth_row_only: int = None,
            reset_pipeline: bool = False,
            **kwargs,
    ) -> Callable[[], Generator[Tuple[Any, pd.DataFrame], None, None]]:
        def get_train_test_batches(batch_size=batch_size, nth_row_only=nth_row_only):
            return self.looper.train_test_iterator(frames, train_test_split_ratio, batch_size, nth_row_only, reset_pipeline)

        if self.history is not None and not reset_pipeline:
            LOG.warning(
                "NOTE retraining the model with a different `train_test_split_ratio` will not affect the data pipeline!\n"
                "If you want to also refit the data pipeline you have to pass `reset_pipeline=True`!"
            )

        self.history = self._fit(get_train_test_batches, **kwargs)

        def y_true_y_hat():
            for prediction in self.predict(frames, include_labels=True):
                yield prediction

        return y_true_y_hat

    def _fit(self, batch_gen: Callable[[], Tuple[BatchCache, ...]], **kwargs) -> Dict[str, np.ndarray]:
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