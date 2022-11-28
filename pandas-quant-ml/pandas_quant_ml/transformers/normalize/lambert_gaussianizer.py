import warnings
from typing import Union, List

import numpy as np
import pandas as pd
import sklearn
from scipy import special, optimize
from scipy.stats import kurtosis

from pandas_df_commons.indexing.decorators import foreach_top_level_row, foreach_column
from pandas_quant_ml._abstract.transfromer import Transformer

_EPS = 1e-6


@foreach_column
@foreach_top_level_row
def ml_lambert_gaussianizer(df: pd.Series, tol: float = 1e-5, max_iter: int = 100, verbose: bool = False):
    df = df.dropna().to_frame()
    lg = LambertGaussianizer(tol, max_iter, verbose)

    return lg.transform(df)


class LambertGaussianizer(sklearn.base.TransformerMixin, Transformer):
    """
    modified version OF https://github.com/gregversteeg/gaussianize/blob/master/gaussianize.py

    Gaussianize data using various methods.
    Conventions
    ----------
    This class is a wrapper that follows sklearn naming/style (e.g. fit(X) to train).
    In this code, x is the input, y is the output. But in the functions outside the class, I follow
    Georg's convention that Y is the input and X is the output (Gaussianized) data.
    Parameters
    ----------

    strategy : str, default='lambert'. Possibilities are 'lambert'[1], 'brute'[2] and 'boxcox'[3].
    tol : float, default = 1e-4
    max_iter : int, default = 100
        Maximum number of iterations to search for correct parameters of Lambert transform.
    Attributes
    ----------
    coefs_ : list of tuples
        For each variable, we have transformation parameters.
        For Lambert, e.g., a tuple consisting of (mu, sigma, delta), corresponding to the parameters of the
        appropriate Lambert transform. Eq. 6 and 8 in the paper below.
    References
    ----------
    [1] Georg M Goerg. The Lambert Way to Gaussianize heavy tailed data with
                        the inverse of Tukey's h transformation as a special case
        Author generously provides code in R: https://cran.r-project.org/web/packages/LambertW/
    [2] Valero Laparra, Gustavo Camps-Valls, and Jesus Malo. Iterative Gaussianization: From ICA to Random Rotations
    [3] Box cox transformation and references: https://en.wikipedia.org/wiki/Power_transform
    """

    def __init__(self, tol: float = 1e-5, max_iter: int = 100, verbose: bool = False):
        super().__init__()
        self.tol = tol
        self.max_iter = max_iter
        self.coefs_ = []  # Store tau for each transformed variable
        self.verbose = verbose

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.coefs_) <= 0:
            self.fit(df.values)

        """Transform new data using a previously learned Gaussianization model."""
        x = self._update_x(df.values)
        if x.shape[1] != len(self.coefs_):
            raise ValueError\
                ("%d variables in test data, but %d variables were in training data." % (x.shape[1], len(self.coefs_)))

        return pd.DataFrame(
            np.array([self._w_t(x_i, tau_i) for x_i, tau_i in zip(x.T, self.coefs_)]).T,
            index=df.index,
            columns=df.columns
        )

    def _inverse(self, df):
        return pd.DataFrame(
            self.inverse_transform(df.values),
            index=df.index,
            columns=df.columns
        )

    def fit(self, x: np.ndarray, y=None):
        """Fit a Gaussianizing transformation to each variable/column in x."""
        # Initialize coefficients again with an empty list.  Otherwise
        # calling .fit() repeatedly will augment previous .coefs_ list.
        self.coefs_ = []
        x = self._update_x(x)

        _get_coef = lambda vec: self._igmm(vec, self.tol, max_iter=self.max_iter)

        for x_i in x.T:
            self.coefs_.append(_get_coef(x_i))

        return self

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Recover original data from Gaussianized data."""
        return np.array([self.__inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.coefs_)]).T

    def _update_x(self, x: Union[np.ndarray, List]) -> np.ndarray:
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            raise ValueError("Data should be a 1-d list of samples to transform or a 2d array with samples as rows.")
        return x

    def _w_d(self, z, delta):
        # Eq. 9
        if delta < _EPS:
            return z
        return np.sign(z) * np.sqrt(np.real(special.lambertw(delta * z ** 2)) / delta)

    def _w_t(self, y, tau):
        # Eq. 8
        return tau[0] + tau[1] * self._w_d((y - tau[0]) / tau[1], tau[2])

    def __inverse(self, x, tau):
        # Eq. 6
        u = (x - tau[0]) / tau[1]
        return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))

    def _igmm(self, y: np.ndarray, tol: float = 1e-6, max_iter: int = 100):
        # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
        if np.std(y) < _EPS:
            return np.mean(y), np.std(y).clip(_EPS), 0
        delta0 = self._delta_init(y)
        tau1 = (np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0)
        for k in range(max_iter):
            tau0 = tau1
            z = (y - tau1[0]) / tau1[1]
            delta1 = self._delta_gmm(z)
            x = tau0[0] + tau1[1] * self._w_d(z, delta1)
            mu1, sigma1 = np.mean(x), np.std(x)
            tau1 = (mu1, sigma1, delta1)

            if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
                break
            else:
                if k == max_iter - 1:
                    warnings.warn("Warning: No convergence after %d iterations. Increase max_iter." % max_iter)
        return tau1

    def _delta_gmm(self, z):
        # Alg. 1, Appendix C
        delta0 = self._delta_init(z)

        def func(q):
            u = self._w_d(z, np.exp(q))
            if not np.all(np.isfinite(u)):
                return 0.
            else:
                k = kurtosis(u, fisher=True, bias=False) ** 2
                if not np.isfinite(k) or k > 1e10:
                    return 1e10
                else:
                    return k

        res = optimize.fmin(func, np.log(delta0), disp=0)
        return np.around(np.exp(res[-1]), 6)

    def _delta_init(self, z):
        gamma = kurtosis(z, fisher=False, bias=False)
        with np.errstate(all='ignore'):
            delta0 = np.clip(1. / 66 * (np.sqrt(66 * gamma - 162.) - 6.), 0.01, 0.48)
        if not np.isfinite(delta0):
            delta0 = 0.01
        return delta0