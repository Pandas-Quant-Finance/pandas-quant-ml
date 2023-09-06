import os
import tempfile

import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.models.keras_model import KerasModel
from pandas_quant_ml.utils.serialize import serialize, deserialize

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from unittest import TestCase

from testing_data.data import get_x_or


class TestKerasModel(TestCase):

    def test_simple_no_test(self):
        df = get_x_or()

        model = KerasModel(
            TrainTestLoop(Select(0, 1), Select("label")),
            keras.Sequential([
                keras.Input(shape=(2,)),
                keras.layers.Dense(1, 'sigmoid'),
            ]),
            loss='binary_crossentropy', optimizer=keras.optimizers.Adam()
        )

        model.fit(df, train_test_split_ratio=1.0, batch_size=6, epochs=3)
        print(model.history)  # {'loss': [0.7795496582984924, 0.7781541347503662, 0.776929497718811]}
        self.assertEquals(len(model.history['loss']), 3)

        _, pdf = next(model.predict(df, include_labels=True))
        print(pdf)
        self.assertGreater(len(pdf), 10)

    def test_train_test_simple(self):
        df = get_x_or()

        model = KerasModel(
            TrainTestLoop(Select(0, 1), Select("label")),
            keras.Sequential([
                keras.Input(shape=(2,)),
                keras.layers.Dense(1, 'sigmoid'),
            ]),
            loss='binary_crossentropy', optimizer=keras.optimizers.Adam()
        )

        model.fit(df, train_test_split_ratio=0.75, batch_size=6, epochs=3)
        print(model.history)  # {'loss': [0.7795496582984924, 0.7781541347503662, 0.776929497718811]}
        self.assertEquals(len(model.history['loss']), 3)

        _, pdf = next(model.predict(df, include_labels=True))
        self.assertEquals((pdf[pdf.columns[-1]] == 'TRAIN').sum(), 38)
        self.assertEquals((pdf[pdf.columns[-1]] == 'TEST').sum(), 12)

        print(pdf)
        self.assertGreater(len(pdf), 10)

    def test_re_train_test_simple(self):
        df = get_x_or()

        model = KerasModel(
            TrainTestLoop(Select(0, 1), Select("label")),
            keras.Sequential([
                keras.Input(shape=(2,)),
                keras.layers.Dense(1, 'sigmoid'),
            ]),
            loss='binary_crossentropy', optimizer=keras.optimizers.Adam()
        )

        model.fit(df, train_test_split_ratio=0.75, batch_size=6, epochs=3)
        print(model.history)  # {'loss': [0.7795496582984924, 0.7781541347503662, 0.776929497718811]}
        self.assertEquals(len(model.history['loss']), 3)

        with self.assertLogs() as captured:
            model.fit(df, train_test_split_ratio=0.9, batch_size=6, epochs=3)

        print(model.history, captured)  # {'loss': [0.7795496582984924, 0.7781541347503662, 0.776929497718811]}
        self.assertEquals(len(model.history['loss']), 6)
        self.assertIn("reset_pipeline=True", ";".join(captured.output))

        with self.assertLogs() as captured:
            model.fit(df, train_test_split_ratio=0.9, batch_size=6, epochs=3, reset_pipeline=True)

        print(model.history, captured)  # {'loss': [0.7795496582984924, 0.7781541347503662, 0.776929497718811]}
        self.assertEquals(len(model.history['loss']), 9)
        self.assertNotIn("reset_pipeline=True", ";".join(captured.output))

        print(model.history_retrained_indexes)
        self.assertListEqual(model.history_retrained_indexes['loss'], [3, 6])

    def test_save_load(self):
        df = get_x_or()

        model = KerasModel(
            TrainTestLoop(Select(0, 1), Select("label")),
            keras.Sequential([
                keras.Input(shape=(2,)),
                keras.layers.Dense(1, 'sigmoid'),
            ]),
            loss='binary_crossentropy', optimizer=keras.optimizers.Adam()
        )

        model.fit(df, train_test_split_ratio=0.8, batch_size=6, epochs=3)
        _, pdf_orig = next(model.predict(df))

        with tempfile.TemporaryDirectory() as td:
            serialize(model, os.path.join(td, "lala"))
            model2 = deserialize(os.path.join(td, "lala"))
            _, pdf_restored = next(model2.predict(df))

        pd.testing.assert_frame_equal(pdf_orig, pdf_restored)
        self.assertEquals(len(model2.history['loss']), 3)

        with self.assertLogs() as captured:
            model2.fit(df, train_test_split_ratio=1.0, batch_size=6, epochs=3)

        self.assertIn("reset_pipeline=True", ";".join(captured.output))