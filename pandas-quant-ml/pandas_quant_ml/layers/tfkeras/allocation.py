import tensorflow as tf


class TargetWeightAllocation(tf.keras.layers.Layer):

    def __init__(self, num_outputs, regularizer: tf.keras.regularizers.Regularizer = None):
        super().__init__()
        self.num_outputs = num_outputs
        self.kernel_regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[1, self.num_outputs],
            initializer="random_normal",
            trainable=True,
            regularizer=self.kernel_regularizer
        )

    def call(self, inputs, *args, **kwargs):
        # NOTE if you allow short positions use a tanh activation for no shorts use a sigmoid activation
        #  in the previous layer
        return self.kernel / tf.reduce_sum(tf.abs(self.kernel))
