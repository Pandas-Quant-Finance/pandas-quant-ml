from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Tuple, Iterable, Generator, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from pandas_quant_ml.data_generators.keras_generator import KerasDataGenerator
from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop


class HyperOptModel(object):

    def __init__(
            self,
            search_space, #Dict[str, hp.*],
            looper: TrainTestLoop,
            model_builder,
            **kwargs
    ):
        super().__init__()
        self.looper = looper
        self.keras_model = model
        self.keras_model_compile_args = kwargs

        self.history = None #:Trials = Trials()

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
        """
        def objective(args):
            # define an objective function -> should be just another model

        # define a search space
        from hyperopt import hp

        # minimize the objective over the space
        from hyperopt import fmin, tpe
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

        print best
        # -> {'a': 1, 'c2': 0.01420615366247227}
        print hyperopt.space_eval(space, best)
        # -> ('case 2', 0.01420615366247227}
        """
        pass

    def predict(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            include_labels: bool = False,
    ) -> Generator[Tuple[Any, pd.DataFrame], None, None]:
        pass

