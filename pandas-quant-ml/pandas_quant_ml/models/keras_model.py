from __future__ import annotations

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
from pandas_quant_ml.models.model import Model
from pandas_quant_ml.utils.batch_cache import BatchCache


class KerasModel(Model):

    def __init__(
            self,
            looper: TrainTestLoop,
            model: tf.keras.Model | Callable[['MetaData'], tf.keras.Model],
            **kwargs
    ):
        super().__init__(looper)

        self.keras_model_provider = lambda md, *args, **kwargs: model if isinstance(model, tf.keras.Model) else model
        self.keras_model_compile_args = kwargs

        self.keras_model: tf.keras.Model = None

        if len(kwargs) > 0 and isinstance(model, tf.keras.Model):
            model.compile(**kwargs)

    def _fit(
            self,
            batch_gen: Callable[[], Tuple[BatchCache, ...]],
            epochs: int = 10,
            workers: int = 1,
            use_multiprocessing: bool = False,
            data_generator_kwargs: Dict = None,
            **kwargs,
    ) -> Dict[str, np.ndarray]:
        train, test = batch_gen()[:2]
        train_it = KerasDataGenerator(train, **(data_generator_kwargs or {}))
        test_it = KerasDataGenerator(test, **(data_generator_kwargs or {}))

        # Only now we know the meta-data and can create the model
        # NOTE if keras_model is not none we continue training on an already trained model
        if self.keras_model is None:
            # Clear clutter from previous TensorFlow graphs.
            tf.keras.backend.clear_session()
            self.keras_model = self.keras_model_provider(self.looper.meta_data)

            if len(self.keras_model_compile_args) > 0:
                self.keras_model.compile(**self.keras_model_compile_args)

        history = deepcopy(self.keras_model.fit(
            train_it,
            validation_data=test_it,
            workers=workers,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=use_multiprocessing,
            **kwargs,
        ).history)

        if "val_acc" in history:
            history["val_accuracy"] = history.pop("val_acc")

        return history

    def get_model_predictor_from_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.keras_model.predict

    def __getstate__(self):
        # don't pickle self.keras_model but save weights into a bin array
        with tempfile.TemporaryDirectory() as temp_dir:
            file = os.path.join(temp_dir, "keras_model")
            self.keras_model.save_weights(file)
            shutil.make_archive(file, 'zip', temp_dir)
            weights = np.fromfile(open(f"{file}.zip", "rb"), dtype=np.dtype('B'))

        return self.looper, self.keras_model_provider, self.keras_model_compile_args, self.history, self.history_retrained_indexes, weights

    def __setstate__(self, state):
        # restore model by using the provider and loading weights
        self.looper, self.keras_model_provider, self.keras_model_compile_args, self.history, self.history_retrained_indexes, weights = state
        self.keras_model = self.keras_model_provider(self.looper.meta_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            file = os.path.join(temp_dir, "keras_model")
            weights.tofile(f"{file}.zip")
            ZipFile(f"{file}.zip").extractall(temp_dir)
            self.keras_model.load_weights(file)
