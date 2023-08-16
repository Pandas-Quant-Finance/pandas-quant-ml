from __future__ import annotations

import array
import os.path
import shutil
import tempfile
from copy import deepcopy
from typing import Any, Dict, Tuple, Iterable, Generator, Callable
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf

from pandas_quant_ml.data_generators.keras_generator import KerasDataGenerator
from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop


class KerasModel(object):

    def __init__(
            self,
            looper: TrainTestLoop,
            model: tf.keras.Model | Callable[['MetaData'], tf.keras.Model],
            **kwargs
    ):
        super().__init__()
        self.looper = looper
        self.keras_model_provider = lambda md, *args, **kwargs: model if isinstance(model, tf.keras.Model) else model
        self.keras_model_compile_args = kwargs

        self.keras_model: tf.keras.Model = None
        self.history: Dict[str, np.ndarray] = None

        if len(kwargs) > 0 and isinstance(model, tf.keras.Model):
            model.compile(**kwargs)

    def fit(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            epochs: int = 10,
            workers: int = 1,
            use_multiprocessing: bool = False,
            keras_fit_kwargs: Dict = None,
            data_generator_kwargs: Dict = None,
            **kwargs,
    ) -> Callable[[], Generator[Tuple[Any, pd.DataFrame], None, None]]:
        train, test = self.looper.train_test_iterator(frames, **kwargs)

        # Only now we know the meta-data and can create the model
        # NOTE if keras_model is not none we continue training on an already trained model
        if self.keras_model is None:
            self.keras_model = self.keras_model_provider(self.looper.meta_data)
            if len(self.keras_model_compile_args) > 0:
                self.keras_model.compile(**self.keras_model_compile_args)

        train_it = KerasDataGenerator(train, **(data_generator_kwargs or {}))
        test_it = KerasDataGenerator(test, **(data_generator_kwargs or {}))

        self.history = deepcopy(self.keras_model.fit(
            train_it,
            validation_data=test_it,
            workers=workers,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=use_multiprocessing,
            **(keras_fit_kwargs or {}),
        ).history)

        if "val_acc" in self.history:
            self.history["val_accuracy"] = self.history.pop("val_acc")

        def y_true_y_hat():
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

    def __getstate__(self):
        # don't pickle self.keras_model but save weights into a bin array
        with tempfile.TemporaryDirectory() as temp_dir:
            file = os.path.join(temp_dir, "keras_model")
            self.keras_model.save_weights(file)
            shutil.make_archive(file, 'zip', temp_dir)
            weights = np.fromfile(open(f"{file}.zip", "rb"), dtype=np.dtype('B'))

        return self.looper, self.keras_model_provider, self.keras_model_compile_args, self.history, weights

    def __setstate__(self, state):
        # restore model by using the provider and loading weights
        self.looper, self.keras_model_provider, self.keras_model_compile_args, self.history, weights = state
        self.keras_model = self.keras_model_provider(self.looper.meta_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            file = os.path.join(temp_dir, "keras_model")
            weights.tofile(f"{file}.zip")
            ZipFile(f"{file}.zip").extractall(temp_dir)
            self.keras_model.load_weights(file)
