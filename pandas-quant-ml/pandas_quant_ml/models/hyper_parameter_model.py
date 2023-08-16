from __future__ import annotations

import logging
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

    def _fit(self, train_test_val: Tuple[BatchCache, ...], model_fit_kwargs: dict = None, **kwargs) -> Dict[str, np.ndarray]:
        if len(train_test_val) < 3:
            LOG.warning(
                "For hyper parameter optimization we expect 3 data sets train, val, test"
                f"but only got {len(train_test_val)}"
            )
            pass

        # optimize for best hyper parameters
        objective = self._create_model_objective(train_test_val[:2], **(model_fit_kwargs or {}))
        self.study.optimize(objective, **kwargs)

        # remember the best objective object
        self.best_objective = self._create_model_objective((train_test_val[0], train_test_val[-1]), **(model_fit_kwargs or {}))
        self.best_objective(self.study.best_trial)

        # return history of hyperparameter trials
        return self._get_result()

    def _create_model_objective(self, train_val: Tuple[BatchCache, ...], **kwargs) -> 'Objective':

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

                history = self.model._fit(train_val, **kwargs)
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
