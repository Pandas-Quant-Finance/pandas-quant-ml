import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from unittest import TestCase

from test_data_generators.data import get_x_or
from pandas_quant_ml.data_generators.keras_generator import KerasDfDataGenerator


class TestKerasGenerator(TestCase):

    def test_simple(self):
        df = get_x_or()
        print(df.tail())

        epochs = 10
        samples = []

        model = keras.Sequential([
            keras.Input(shape=(2,)),
            keras.layers.Dense(1, 'sigmoid'),
        ])
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

        # Train model on dataset
        model.fit_generator(
            generator=KerasDfDataGenerator(df, [0, 1], 'label', batch_size=6, on_get_batch=lambda i, x, y: samples.append(len(x))),
            validation_data=None,
            use_multiprocessing=True,
            workers=2,
            epochs=epochs,
            shuffle=False
        )

        print(samples)
        print(sum(samples) / epochs, len(df))
        pass

    def test_lookback_window(self):
        pass


