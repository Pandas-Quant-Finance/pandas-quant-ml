from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.generic.cleansing import CleanNaN
from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.data_transformers.generic.shift import PredictDateTimePeriods
from pandas_quant_ml.data_transformers.generic.windowing import MovingWindow
from pandas_quant_ml.models.keras_model import KerasModel
from tensorflow import keras


class TestDivergingLabelPredictionShapes(TestCase):

    def test_more_labels(self):
        df = pd.DataFrame(np.random.normal(0, 0.2, (200, 2)), columns=["A", "B"])
        looper = TrainTestLoop(
            Select("A", "B"),
            Select("A", "B") \
                >> PredictDateTimePeriods(5) \
                >> CleanNaN() \
                >> MovingWindow(5)
        )

        model = KerasModel(
            looper,
            keras.Sequential([
                keras.Input(shape=(2,)),
                keras.layers.Dense(2, 'sigmoid'),
            ]),
            loss='mse', optimizer=keras.optimizers.Adam()
        )

        model.keras_model = model.keras_model_provider(None)
        _ = looper.train_test_iterator(df)
        _, pred = next(model.predict(df))

        print(pred.iloc[-2:])
        self.assertEquals(pred.shape, (200, 2))

    def test_more_predictions(self):
        df = pd.DataFrame(np.random.normal(0, 0.2, (200, 2)), columns=["A", "B"])
        looper = TrainTestLoop(
            Select("A", "B"),
            Select("A", "B"),
        )

        model = KerasModel(
            looper,
            keras.Sequential([
                keras.Input(shape=(2,)),
                keras.layers.Dense(10, 'sigmoid'),
                keras.layers.Reshape((5, 2)) # FIXME we can only simulate a bigger than batch array using pytorch
            ]),
            loss='mse', optimizer=keras.optimizers.Adam()
        )

        model.keras_model = model.keras_model_provider(None)
        _ = looper.train_test_iterator(df)
        _, pred = next(model.predict(df))

        print(pred.iloc[-5:])
        #self.assertEquals(pred.shape, (200, 2))
