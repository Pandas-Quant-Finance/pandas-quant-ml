import tensorflow as tf


class SharpeLoss(tf.keras.losses.Loss):

    def __init__(self, output_size: int = 1, scaling_factor: float = 252.0, sortino: bool = False):
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        self.scaling_factor = scaling_factor
        super().__init__()

    def call(self, y_true, weights):
        """
        Args:
            y_true: asset returns
            weights: predicted weights

        Returns:

        """
        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        std = tf.sqrt(tf.reduce_mean((captured_returns - mean_returns) ** 2) + 1e-9)
        # if sortino: std = tf.sqrt(tf.reduce_mean(tf.minimum(captured_returns - mean_returns, 0) ** 2) + 1e-9)
        return -(
            (mean_returns / std)
            * tf.sqrt(self.scaling_factor)
        )
