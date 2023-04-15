from typing import Iterable, Callable

import numpy as np


class ModelScorer(object):

    def __init__(
            self,
            data_generator: Iterable,
            predictor: Callable,
    ):
        super().__init__()
        self.data_generator = data_generator
        self.predictor = predictor

    def score(self):
        res = [(y, self.predictor(x).reshape(y.shape)) for x, y in self.data_generator if len(x) > 0]

        if len(res) <= 0: return None
        y = np.concatenate([y[0] for y in res], axis=0)
        y_hat = np.concatenate([y[1] for y in res], axis=0)
        residual = y_hat - y  # 1 - 0.8 = 0.2 vs 0.8 - 1 = -0.2
        return y, y_hat, residual

