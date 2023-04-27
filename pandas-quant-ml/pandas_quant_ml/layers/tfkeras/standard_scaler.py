import tensorflow as tf


class StandardScalerLayer(tf.keras.layers.Layer):

    def __init__(self, axis=-1, scaling=0.3):
        super().__init__()
        self.scaling = scaling
        self.axis = axis

    def call(self, x, *args, **kwargs):
        x -= tf.reduce_mean(x, axis=self.axis, keepdims=True)
        std = tf.math.reduce_std(x, self.axis, keepdims=True)
        return tf.math.divide_no_nan(x, std) * self.scaling
