import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from unittest import TestCase

from test_data_generators.data import get_x_or
from pandas_quant_ml.data_generators.keras_generator import KerasDfDataGenerator
from pandas_quant_ml.model_scoring.model_scorer import ModelScorer


class TestKerasGenerator(TestCase):

    def test_concept(self):
        df = get_x_or()
        it = KerasDfDataGenerator(df, [0, 1], 'label', batch_size=6)
        samples = 0
        for epoch in range(2):
            for i in range(len(it)):
               samples += len(it[i][0])

        self.assertEquals(len(df) * 2, samples)

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
            workers=2,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=False,
        )

        # print(samples)
        self.assertGreater(sum(samples) / epochs, len(df))
        residuals = ModelScorer(
            KerasDfDataGenerator(df, [0, 1], 'label', batch_size=6, shuffle=False),
            model.predict
        ).score()
        print(residuals)

    def test_lookback_window(self):
        df = get_x_or()
        print(df.tail())

        epochs = 10
        samples = []

        model = keras.Sequential([
            keras.Input(shape=(4, 2,)),
            keras.layers.Flatten(),
            keras.layers.Dense(1, 'sigmoid'),
        ])
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

        # Train model on dataset
        model.fit_generator(
            generator=KerasDfDataGenerator(df, [0, 1], 'label', batch_size=6, feature_look_back_window=4, on_get_batch=lambda i, x, y: samples.append(len(x))),
            validation_data=None,
            workers=2,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=False,
        )

        print(samples)

