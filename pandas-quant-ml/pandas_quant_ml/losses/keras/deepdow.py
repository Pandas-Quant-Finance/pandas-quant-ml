from typing import Literal

# NOTE THAT THIS IS A COPY OF deepdow.losses WHICH IS FOR pytorch
# Also note, in keras the loss function signature is loss_fn(y_true, y_pred)  while in pytorch it is my_loss(output, target)

import tensorflow as tf
import tensorflow_probability as tfp
from types import MethodType
#from .layers import CovarianceMatrix

import torch


def covariance(x: tf.Tensor, y: tf.Tensor, sample_axis=0, event_axis=None):
    return tfp.stats.covariance(x, y, sample_axis, event_axis)


def log2simple(x: tf.Tensor):
    """Turn log returns into simple returns.

    r_simple = exp(r_log) - 1.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of any shape where each entry represents a logarithmic return.

    Returns
    -------
    torch.Tensor
        Simple returns.

    """
    return tf.math.exp(x) - 1


def simple2log(x: tf.Tensor):
    """Turn simple returns into log returns.

    r_log = ln(r_simple + 1).

    Parameters
    ----------
    x : torch.Tensor
        Tensor of any shape where each entry represents a simple return.

    Returns
    -------
    torch.Tensor
        Logarithmic returns.

    """
    return tf.math.log(x + 1.0)


def portfolio_returns(
        y: tf.Tensor, weights: tf.Tensor, input_type: Literal['log', 'simple'] = 'log', output_type: Literal['log', 'simple'] = "simple", rebalance: bool = False
) -> tf.Tensor:
    """Compute portfolio returns.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape (n_samples, n_assets) representing the simple buy and hold strategy over the horizon.

    y : torch.Tensor
        Tensor of shape (n_samples, horizon, n_assets) representing single period non-cumulative returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    rebalance : bool
        If True, each timestep the weights are adjusted to be equal to the original ones. Note that
        this assumes that we tinker with the portfolio. If False, the portfolio evolves untouched.

    Returns
    -------
    portfolio_returns : torch.Tensor
        Of shape (n_samples, horizon) representing per timestep portfolio returns.

    """
    if input_type == "log":
        simple_returns = log2simple(y)

    elif input_type == "simple":
        simple_returns = y

    else:
        raise ValueError("Unsupported input type: {}".format(input_type))

    weights = tf.reshape(tf.repeat(weights, simple_returns.shape[1], axis=0), (-1, *simple_returns.shape[1:]))

    if not rebalance:
        weights_unscaled = tf.math.cumprod(1 + simple_returns, axis=1)[:, :-1, :] * weights[:, 1:, :]
        weights = tf.concat([weights[:, :1, :], weights_unscaled / tf.reduce_sum(weights_unscaled, axis=2, keepdims=True)], axis=1)

    out = tf.reduce_sum(simple_returns * weights, axis=-1)

    if output_type == "log":
        return simple2log(out)

    elif output_type == "simple":
        return out

    else:
        raise ValueError("Unsupported output type: {}".format(output_type))


def portfolio_cumulative_returns(
    y: tf.Tensor, weights: tf.Tensor, input_type: Literal['log', 'simple'] = 'log', output_type: Literal['log', 'simple'] = "simple", rebalance: bool = False
):
    """Compute cumulative portfolio returns.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the log return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    rebalance : bool
        If True, each timestep the weights are adjusted to be equal to be equal to the original ones. Note that
        this assumes that we tinker with the portfolio. If False, the portfolio evolves untouched.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples, horizon)`.

    """
    prets = portfolio_returns(
        y,
        weights,
        input_type=input_type,
        output_type="log",
        rebalance=rebalance,
    )
    log_prets = tf.math.cumsum(
        prets, axis=1
    )  # we can aggregate log returns over time by sum

    if output_type == "log":
        return log_prets

    elif output_type == "simple":
        return log2simple(log_prets)

    else:
        raise ValueError("Unsupported output type: {}".format(output_type))


class Loss:
    """Parent class for all losses.

    Additionally it implement +, -, * and / operation between losses.
    """

    def _call(self, weights, y):
        raise NotImplementedError()

    def _repr(self):
        raise NotImplementedError()

    def __call__(self, weights, y):
        """Compute loss.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, horizon, n_assets)` representing ground truth labels
            over the `horizon` of steps. The idea is that the channel dimensions can be given a specific meaning
            in the constructor.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample loss.

        """
        return self._call(weights, y)

    def __repr__(self):
        """Generate representation string.

        The goal is two generate a string `s` that we can `eval(s)` to instantiate the loss.
        """
        return self._repr()

    def __add__(self, other):
        """Add two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the sum of `self` and `other`. If a number
            then create a new loss that is equal to `self` plus a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the addition operation.
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) + other(weights, y),
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "{} + {}".format(
                    self.__repr__(), other.__repr__()
                ),
                new_instance,
            )

            return new_instance

        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) + other, new_instance
            )
            new_instance._repr = MethodType(
                lambda inst: "{} + {}".format(self.__repr__(), other),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(other)))

    def __radd__(self, other):
        """Add two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the sum of `self` and `other`. If a number
            then create a new loss that is equal to `self` plus a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the addition operation.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the product of `self` and `other`. If a
            number then create a new loss that is equal to `self` times a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the multiplication operation.
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) * other(weights, y),
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "{} * {}".format(
                    self.__repr__(), other.__repr__()
                ),
                new_instance,
            )

            return new_instance

        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) * other, new_instance
            )
            new_instance._repr = MethodType(
                lambda inst: "{} * {}".format(self.__repr__(), other),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(other)))

    def __rmul__(self, other):
        """Multiply two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the product of `self` and `other`. If a
            number then create a new loss that is equal to `self` times a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the multiplication operation.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the ratio of `self` and `other`. If a
            number then create a new loss that is equal to `self` divided a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the division operation.
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) / other(weights, y),
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "{} / {}".format(
                    self.__repr__(), other.__repr__()
                ),
                new_instance,
            )

            return new_instance

        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError()

            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) / other, new_instance
            )
            new_instance._repr = MethodType(
                lambda inst: "{} / {}".format(self.__repr__(), other),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(other)))

    def __pow__(self, power):
        """Put a loss to a power.

        Parameters
        ----------
        power : int or float
            Number representing the exponent

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the `self ** power`.
        """
        if isinstance(power, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) ** power,
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "({}) ** {}".format(self.__repr__(), power),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(power)))


# FIXME -- W STOPPED HERE
class CumulativeReturn(Loss):
    """Negative cumulative returns.

    Parameters
    ----------
    returns_channel : int
        Which channel of the `y` target represents returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.
    """

    def __init__(self, returns_channel=0, input_type="log"):
        self.returns_channel = returns_channel
        self.input_type = input_type

    def __call__(self, weights, y):
        """Compute negative simple cumulative returns.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative simple cumulative returns.

        """
        crets = portfolio_cumulative_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type="simple",
        )

        return -crets[:, -1]

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}')".format(
            self.__class__.__name__, self.returns_channel, self.input_type
        )


class SharpeRatio(Loss):
    """Negative Sharpe ratio.

    Parameters
    ----------
    rf : float
        Risk-free rate.

    returns_channel : int
        Which channel of the `y` target represents returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    eps : float
        Additional constant added to the denominator to avoid division by zero.
    """

    def __init__(
        self,
        rf=0,
        returns_channel=0,
        input_type="log",
        output_type="simple",
        eps=1e-4,
    ):
        self.rf = rf
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.eps = eps

    def __call__(self, y, weights):
        """Compute negative sharpe ratio.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative sharpe ratio.

        """
        prets = portfolio_returns(
            y[:, self.returns_channel, ...],
            weights,
            input_type=self.input_type,
            output_type=self.output_type,
        )

        mean = tf.reduce_mean(prets, axis=1) - self.rf
        #std = tf.math.reduce_std(prets, axis=1) + self.eps
        std = tf.math.sqrt(tf.experimental.numpy.var(prets, dtype=mean.dtype, axis=1, ddof=1)) + self.eps
        return -mean / std

    def __repr__(self):
        """Generate representation string."""
        return "{}(rf={}, returns_channel={}, input_type='{}', output_type='{}', eps={})".format(
            self.__class__.__name__,
            self.rf,
            self.returns_channel,
            self.input_type,
            self.output_type,
            self.eps,
        )