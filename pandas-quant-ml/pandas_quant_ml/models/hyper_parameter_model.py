from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, Tuple, Iterable, Generator, Callable

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.trial import TrialState, BaseTrial

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.models.model import Model
from pandas_quant_ml.utils.batch_cache import BatchCache

LOG = logging.getLogger(__name__)


class OptunaModel(Model):

    def __init__(
            self,
            looper: TrainTestLoop,
            study: optuna.Study,
            metric: str,
            search_space: Dict[str, Callable[[BaseTrial], Any]],
            model_factory: Callable[[Any, ...], Model],
    ):
        super().__init__(looper)
        self.study = study
        self.search_space = search_space
        self.metric = metric
        self.model_factory = model_factory

        self.best_objective = None

    def _fit(
            self,
            batch_gen: Callable[[], Tuple[BatchCache, ...]],
            model_fit_kwargs: dict = None,
            **kwargs
    ) -> Dict[str, np.ndarray]:
        if not (isinstance(self.looper._train_test_split_ratio, tuple) and len(self.looper._train_test_split_ratio) > 1):
            LOG.warning(
                "For hyper parameter optimization we expect 3 data sets train, val, test"
                "and thus a train_test_split_ratio tuple(%-train, %-val) "
                f"but only got {self.looper._train_test_split_ratio}"
            )
            pass

        # optimize for best hyper parameters
        objective = self._create_model_objective(batch_gen, **(model_fit_kwargs or {}))
        self.study.optimize(objective, **kwargs)

        # remember the best objective object
        self.best_objective = objective
        self.best_objective(self.study.best_trial)

        # return history of hyperparameter trials
        return self._get_result()

    def _create_model_objective(self, batch_gen, **kwargs) -> 'Objective':

        class Objective(object):

            def __init__(self, model_factory, search_space, monitor):
                self.model_factory = model_factory
                self.search_space = search_space
                self.monitor = monitor

                self.hyper_parameters = None
                self.model = None

            def __call__(self, trial):
                self.hyper_parameters = {p: v(trial) for p, v in self.search_space.items()}
                self.model = self.model_factory(**self.hyper_parameters)

                if 'batch_size' in self.hyper_parameters:
                    _batch_gen = partial(batch_gen, batch_size=self.hyper_parameters['batch_size'])
                else:
                    # if we do not use hyperparameters which affect the pipeline we can cache the data sets
                    train_val_test = batch_gen()
                    _batch_gen = lambda **kwargs: train_val_test

                history = self.model._fit(_batch_gen, **kwargs)
                return history[self.monitor][-1]

        return Objective(
            lambda **hp: self.model_factory(self.looper, **hp),
            self.search_space,
            self.metric
        )

    def _get_result(self) -> Dict:
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        best_trail = self.study.best_trial

        return {
            "Number of finished trials": len(self.study.trials),
            "Number of pruned trials": len(pruned_trials),
            "Number of complete trials": len(complete_trials),
            "Best value": best_trail.value,
            "Best parameters": best_trail.params,
            "metric": np.array([t.value for t in complete_trials])
        }

    def get_model_predictor_from_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.best_objective.model.get_model_predictor_from_numpy()

    @property
    def best_model(self):
        assert self.best_objective is not None, "Needs to be fitted first"
        return self.best_objective.model