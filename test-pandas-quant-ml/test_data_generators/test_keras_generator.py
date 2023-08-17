import os
import unittest
from datetime import datetime

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.generic.selection import Select

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from unittest import TestCase

from testing_data.data import get_x_or
from pandas_quant_ml.data_generators.keras_generator import KerasDataGenerator


class TestKerasGenerator(TestCase):

    def test_concept(self):
        df = get_x_or()
        looper = TrainTestLoop(Select(0, 1), Select("label"))
        train, test = looper.train_test_iterator(df, train_test_split_ratio=1.0, batch_size=5)
        it = KerasDataGenerator(train)

        samples = 0
        for epoch in range(2):
            for i in range(len(it)):
               samples += len(it[i][0])

        self.assertEquals(len(df) * 2, samples)

    def test_simple(self):
        start_at = datetime.now()
        df = get_x_or()
        epochs = 10
        samples = []

        looper = TrainTestLoop(Select(0, 1), Select("label"))
        train, test = looper.train_test_iterator(df, train_test_split_ratio=1.0, batch_size=6)
        it = KerasDataGenerator(train, on_batch_end=lambda _, i, x, y, w: samples.append(x.shape[0]), on_epoch_end=print)

        model = keras.Sequential([
            keras.Input(shape=(2,)),
            keras.layers.Dense(1, 'sigmoid'),
        ])
        print("\ncompile model after ", (datetime.now() - start_at).seconds, "sec")
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())

        # Train model on dataset
        print("\nstart training after ", (datetime.now() - start_at).seconds, "sec")
        model.fit_generator(
            generator=it,
            validation_data=None,
            workers=2,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=False,
        )

        print("\nstart evaluation after ", (datetime.now() - start_at).seconds, "sec")
        # FIXME bring back this stuff with the ModelScorer
        # print(samples)
        self.assertGreater(sum(samples) / epochs, len(df))
        #residuals = ModelScorer(
        #    KerasDfDataGenerator(df, [0, 1], 'label', batch_size=6, shuffle=False),
        #    model.predict
        #).score()
        #print(residuals)

    @unittest.skip('lookback window is obsolete')
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

