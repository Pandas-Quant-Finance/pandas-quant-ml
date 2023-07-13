from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Tuple, Iterable, Generator, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from pandas_quant_ml.data_generators.keras_generator import KerasDataGenerator
from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop


class KerasModel(object):

    def __init__(
            self,
            looper: TrainTestLoop,
            model: tf.keras.Model,
            **kwargs
    ):
        super().__init__()
        self.looper = looper
        self.keras_model = model

        self.history: Dict[str, np.ndarray] = None

        if len(kwargs) > 0:
            model.compile(**kwargs)

    def fit(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            epochs: int = 10,
            workers: int = 1,
            use_multiprocessing: bool = False,
            keras_fit_kwargs: Dict = None,
            **kwargs
    ) -> Callable[[], Generator[Tuple[Any, pd.DataFrame], None, None]]:
        train, test = self.looper.train_test_iterator(frames)
        train_it = KerasDataGenerator(train, **kwargs)
        test_it = KerasDataGenerator(test, **kwargs)

        self.history = deepcopy(self.keras_model.fit(
            train_it,
            validation_data=test_it,
            workers=workers,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=use_multiprocessing,
            **(keras_fit_kwargs or {}),
        ).history)

        def y_true_y_hat():
            # TODO split into train and test frames
            for prediction in self.predict(frames, include_labels=True):
                yield prediction

        return y_true_y_hat

    def predict(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            include_labels: bool = False,
    ) -> Generator[Tuple[Any, pd.DataFrame], None, None]:
        for prediction in self.looper.inference_generator(frames, self.keras_model.predict, include_labels):
            yield prediction


