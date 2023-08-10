import os

import numpy as np
import numpy.testing
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU

import tensorflow as tf
import torch
from unittest import TestCase

from pandas_quant_ml.losses.keras.deepdow import portfolio_returns as k_portfolio_returns
from pandas_quant_ml.losses.pytorch.deepdow import portfolio_returns as t_portfolio_returns
from pandas_quant_ml.losses.keras.deepdow import portfolio_cumulative_returns as k_portfolio_cumulative_returns
from pandas_quant_ml.losses.pytorch.deepdow import portfolio_cumulative_returns as t_portfolio_cumulative_returns


@pytest.mark.parametrize("loss", [
    (k_portfolio_returns, ), (k_portfolio_cumulative_returns, )
])
def test_portfolio_returns(loss):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(11),
        tf.keras.layers.Dense(11, activation='sigmoid'),
        tf.keras.layers.Softmax(),
    ])

    model.compile(loss=loss, optimizer='SGD')
    print(model.predict(tf.random.normal([1, 11])))
    model.fit(tf.random.normal([1, 11]), tf.random.normal([1, 3, 11]), epochs=3)

@pytest.mark.parametrize("input_type,output_type,rebalance", [
    ('log', 'simple', False),
    ('simple', 'simple', False),
    ('simple', 'log', False),
    ('log', 'simple', True),
    ('simple', 'simple', True),
    ('simple', 'log', True),
])
def test_portfolio_returns_torch_keras(input_type, output_type, rebalance):
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(11),])

    x = np.random.normal(0, 0.2, (1, 11))
    y = np.random.normal(0, 0.2, (1, 10, 11))

    numpy.testing.assert_array_almost_equal(
        k_portfolio_returns(tf.convert_to_tensor(y, dtype=tf.float32), model(tf.convert_to_tensor(x, dtype=tf.float32)), input_type, output_type, rebalance),
        t_portfolio_returns(torch.from_numpy(model.predict(tf.convert_to_tensor(x, dtype=tf.float32))), torch.from_numpy(y.astype('float32')), input_type, output_type, rebalance),
        decimal=4
    )

@pytest.mark.parametrize("input_type,output_type,rebalance", [
    ('log', 'simple', False),
    ('simple', 'simple', False),
    ('simple', 'log', False),
    ('log', 'simple', True),
    ('simple', 'simple', True),
    ('simple', 'log', True),
])
def test_portfolio_cumreturns_torch_keras(input_type, output_type, rebalance):
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(11),])

    x = np.random.normal(0, 0.2, (1, 11))
    y = np.random.normal(0, 0.2, (1, 10, 11))

    numpy.testing.assert_array_almost_equal(
        k_portfolio_cumulative_returns(tf.convert_to_tensor(y, dtype=tf.float32), model(tf.convert_to_tensor(x, dtype=tf.float32)), input_type, output_type, rebalance),
        t_portfolio_cumulative_returns(torch.from_numpy(model.predict(tf.convert_to_tensor(x, dtype=tf.float32))), torch.from_numpy(y.astype('float32')), input_type, output_type, rebalance),
        decimal=4
    )
