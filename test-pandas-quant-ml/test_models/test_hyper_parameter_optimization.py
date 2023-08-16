import os
import tempfile

import optuna
import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.models.hyper_parameter_model import OptunaModel
from pandas_quant_ml.models.keras_model import KerasModel
from pandas_quant_ml.utils.serialize import serialize, deserialize

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from unittest import TestCase

from test_data_generators.data import get_x_or


class TestOptunaModel(TestCase):

    def test_hyper_parameters_keras(self):
        df = get_x_or()

        def model_factory(looper, **hyper_parameters):
            return KerasModel(
                looper,
                keras.Sequential([
                    keras.Input(shape=(2,)),
                    keras.layers.Dense(hyper_parameters['nodes'], 'sigmoid'),
                    keras.layers.Dense(1, 'sigmoid'),
                ]),
                loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hyper_parameters['learning_rate'])
            )

        model = OptunaModel(
            TrainTestLoop(Select(0, 1), Select("label"), train_test_split_ratio=0.5, batch_size=6), # FIXME use train, val test split
            optuna.create_study(
                direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
            ),
            'val_loss',
            {
                "learning_rate": lambda trial: trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "nodes": lambda trial: trial.suggest_categorical("nodes", [2, 5, 10]),
            },
            model_factory
        )

        model.fit(df, n_trials=5, timeout=600)
        print(model.history)
        self.assertEquals(len(model.history['metric']), 5)

        _, pdf = next(model.predict(df, include_labels=True))
        print(pdf)
        self.assertEquals((pdf[pdf.columns[-1]] == 'TRAIN').sum(), 25)
        self.assertEquals((pdf[pdf.columns[-1]] == 'TEST').sum(), 25)
        # FIXME self.assertEquals((pdf[pdf.columns[-1]] == 'VAL').sum(), ??)
