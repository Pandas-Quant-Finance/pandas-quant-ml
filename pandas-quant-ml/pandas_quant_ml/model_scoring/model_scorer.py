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
        res = [y - self.predictor(x).reshape(y.shape) for x, y in self.data_generator]

        if len(res) <= 0: return None

        if res[0].ndim == 1:
            return np.hstack(res)
        else:
            return np.stack(res, axis=0)
