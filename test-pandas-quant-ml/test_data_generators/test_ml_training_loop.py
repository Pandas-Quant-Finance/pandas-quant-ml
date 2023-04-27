from unittest import TestCase

import pandas as pd
import numpy as  np
from pandas_quant_ml.data_generators.ml_traning_loop import TrainingLoop

DF = pd.DataFrame(np.random.random((30, 2)), index=pd.MultiIndex.from_product([["A", "B"], range(15)]))


class TestMlTrainingLoop(TestCase):

    def test_training_loop_single_simple(self):
        df = DF.loc["A"].copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in TrainingLoop(df, [0, 1], "label", 3)])
        np.testing.assert_almost_equal(df["label"].values, res)

    def test_training_loop_simple(self):
        df = DF.copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in TrainingLoop(df, [0, 1], "label", 3)])
        np.testing.assert_array_equal(df["label"].values, res)

    def test_training_loop_window(self):
        df = DF.copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in TrainingLoop(df, [0, 1], "label", 4, 3)])
        np.testing.assert_array_equal(
            np.hstack([df.loc["A"]["label"][2:].values, df.loc["B"]["label"][2:].values]),
            res
        )

    def test_training_loop_simple_shuffled(self):
        df = DF.copy()
        df["label"] = df.index

        res = list(TrainingLoop(df, [0, 1], "label", 4, shuffle=True))
        for f, i in res:
            np.testing.assert_almost_equal(df.loc[i][[0, 1]].values, f)

    def test_training_loop_window_shuffled(self):
        df = DF.copy()
        df["label"] = df.index

        res = list(TrainingLoop(df, [0, 1], "label", 4, 3, shuffle=True))
        for f, i in res:
            np.testing.assert_almost_equal(df.loc[i][[0, 1]].values, f[:, -1, :])

    def test_training_loop_2window_shuffled(self):
        df = DF.copy()
        df["label"] = df.index

        res = list(TrainingLoop(df, [0, 1], [0, 1], 4, 3, label_look_back_window=3, shuffle=True))
        for f, l in res:
            np.testing.assert_almost_equal(f, l)

    def test_weights_and_transformer(self):
        df = DF.copy()
        df["label"] = df.index.get_level_values(1)
        df["weight"] = range(len(df))

        features_lables_weights = list(
            TrainingLoop(
                df, [0, 1], "label", 4, label_weight_columns="weight",
                label_transformer=lambda l: l *10
            )
        )
        features = np.concatenate([flw[0] for flw in features_lables_weights])
        labels = np.concatenate([flw[1] for flw in features_lables_weights])
        weights = np.concatenate([flw[2] for flw in features_lables_weights])

        np.testing.assert_almost_equal(df[[0, 1]].values, features)
        np.testing.assert_almost_equal(df["label"].values * 10, labels)
        np.testing.assert_almost_equal(df["weight"].values, weights)

    def test_multiple_epochs(self):
        df = DF.copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in TrainingLoop(df, [0, 1], "label", 3, epochs=2)])
        np.testing.assert_array_equal(np.concatenate([df["label"].values] * 2, axis=0), res)

    def test_item_access(self):
        df = DF.copy()
        df["label"] = df.index

        tl = TrainingLoop(df, [0, 1], "label", 3)
        res = np.concatenate([tl[i][1] for i in list(range(len(tl))) * 2])
        np.testing.assert_array_equal(np.concatenate([df["label"].values] * 2, axis=0), res)
