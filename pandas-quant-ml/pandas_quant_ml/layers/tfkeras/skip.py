from __future__ import annotations
import tensorflow as tf


class SkipConnection(tf.keras.layers.Layer):

    def __init__(self, sequential: list, sequential_residual = None, ):
        super().__init__()
        self.sequential = tf.keras.Sequential(sequential)
        self.sequential_res = tf.keras.Sequential(sequential_residual) if sequential_residual else None

    def build(self, input_shape):
        self.sequential.build(input_shape)
        if self.sequential_res is not None:
            self.sequential_res.build(input_shape)
        else:
            self.sequential_res = lambda x, *args: x

    def call(self, inputs, training=None, mask=None):
        return self.sequential(inputs, training, mask) + self.sequential(inputs, training, mask)

