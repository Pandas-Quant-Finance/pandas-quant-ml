from __future__ import annotations

import os.path
import shutil
import tempfile
import torch
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Tuple, Callable
from zipfile import ZipFile

import lightning.pytorch as pl
import numpy as np
import tensorflow as tf

from pandas_quant_ml.data_generators.pytorch_generator import TorchDataGenerator
from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.models.model import Model
from pandas_quant_ml.utils.batch_cache import BatchCache


class LightningModel(Model):

    def __init__(
            self,
            looper: TrainTestLoop,
            model: pl.LightningModule | Callable[['MetaData'], pl.LightningModule],
            dtype: torch.Type | Tuple[torch.Type, ...] = torch.float32,
    ):
        super().__init__(looper)

        self.lightning_model_provider = lambda md, *args, **kwargs: model if isinstance(model, tf.keras.Model) else model
        self.lightning_model: pl.LightningModule = None
        self.dtype = dtype

    def _fit(
            self,
            batch_gen: Callable[[], Tuple[BatchCache, ...]],
            trainer_fit_kwargs: Dict = None,
            data_generator_kwargs: Dict = None,
            **kwargs,
    ) -> Dict[str, np.ndarray]:
        train, test = batch_gen()[:2]
        train_it = TorchDataGenerator(train, dtype=self.dtype, **(data_generator_kwargs or {}))
        test_it = TorchDataGenerator(test, dtype=self.dtype, **(data_generator_kwargs or {}))

        # Only now we know the meta-data and can create the model
        # NOTE if keras_model is not none we continue training on an already trained model
        if self.lightning_model is None:
            self.lightning_model = self.lightning_model_provider(self.looper.meta_data)

        if 'callbacks' in kwargs:
            if not isinstance(kwargs['callbacks'], list):
                kwargs['callbacks'] = [kwargs['callbacks']]
        else:
            kwargs['callbacks'] = []

        history = _MetricTracker()
        kwargs['callbacks'].append(history)
        # kwargs['log_every_n_steps'] = 1

        trainer = pl.Trainer(**kwargs)
        trainer.fit(self.lightning_model, train_it, test_it, **(trainer_fit_kwargs or {}))

        return history.history

    def get_model_predictor_from_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        def predictor(x):
            with torch.no_grad():
                return self\
                    .lightning_model(torch.from_numpy(x).type(self.dtype[0] if isinstance(self.dtype, tuple) else self.dtype))\
                    .numpy()

        return predictor


class _MetricTracker(pl.Callback):

    def __init__(self):
        self._current_epoch = defaultdict(list)
        self.hist = defaultdict(list)

    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        outputs = trainer.logged_metrics
        for o, val in outputs.items():
            self._current_epoch[o].append(val.detach().numpy())

    def on_test_batch_end(self, trainer, module, *args, **kwargs):
        outputs = trainer.logged_metrics
        for o, val in outputs.items():
            if o not in self._current_epoch:
                self._current_epoch[o].append(val.detach().numpy())

    def on_validation_batch_end(self, trainer, module, *args, **kwargs):
        self.on_test_batch_end(trainer, module, *args, **kwargs)

    def on_train_epoch_end(self, trainer, module, *args, **kwargs):
        for k, v in self._current_epoch.items():
            self.hist[k].append(np.array(v))

    @property
    def history(self):
        return {k: [v.mean() for v in vs] for k, vs in self.hist.items()}
