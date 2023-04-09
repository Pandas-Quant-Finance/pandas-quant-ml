from unittest import TestCase

import pandas as pd
import numpy as  np
from pandas_quant_ml.data_generators.ml_traning_loop import batch_generator, training_loop

DF = pd.DataFrame(np.random.random((30, 2)), index=pd.MultiIndex.from_product([["A", "B"], range(15)]))


class TestMlTrainingLoop(TestCase):

    def test_batch_simple(self):
        batches = [b for _, b in batch_generator(DF, 4, )]
        res = np.vstack(batches)
        print(res)

        self.assertGreater(len(batches), 2)
        self.assertLess(len(batches[-1]), 4)
        np.testing.assert_almost_equal(DF.values, res, 5)

    def test_batch_window_simple(self):
        batches = [b for _, b in batch_generator(DF, 4, 3)]

        self.assertGreater(len(batches), 2)
        self.assertLess(len(batches[-1]), 4)
        np.testing.assert_almost_equal(DF[:3].values, batches[0][0], 5)
        np.testing.assert_almost_equal(DF[-3:].values, batches[-1][-1], 5)

    def test_batch_shuffled(self):
        batches1 = [b for _, b in batch_generator(DF, 4, shuffle=False)]
        batches2 = [b for _, b in batch_generator(DF, 4, shuffle=True)]
        self.assertEqual(len(batches1), len(batches2))

    def test_batch_window_shuffled(self):
        batches1 = [b for _, b in batch_generator(DF, 4, 3, shuffle=False)]
        batches2 = [b for _, b in batch_generator(DF, 4, 3, shuffle=True)]

        self.assertEqual(len(batches1), len(batches2))
        for i in range(len(batches1)):
            self.assertEqual(batches1[i].shape, batches2[i].shape, str(i))
            self.assertFalse(np.array_equal(batches1[i], batches2[i]), str(i))

    def test_training_loop_single_simple(self):
        df = DF.loc["A"].copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in training_loop(df, [0, 1], "label", 3)])
        np.testing.assert_almost_equal(df["label"].values, res)

    def test_training_loop_simple(self):
        df = DF.copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in training_loop(df, [0, 1], "label", 3)])
        np.testing.assert_array_equal(df["label"].values, res)

    def test_training_loop_window(self):
        df = DF.copy()
        df["label"] = df.index

        res = np.hstack([l for f, l in training_loop(df, [0, 1], "label", 4, 3)])
        np.testing.assert_array_equal(
            np.hstack([df.loc["A"]["label"][2:].values, df.loc["B"]["label"][2:].values]),
            res
        )

    def test_training_loop_simple_shuffled(self):
        df = DF.copy()
        df["label"] = df.index

        res = list(training_loop(df, [0, 1], "label", 4, shuffle=True))
        for f, i in res:
            np.testing.assert_almost_equal(df.loc[i][[0, 1]].values, f)

    def test_training_loop_window_shuffled(self):
        df = DF.copy()
        df["label"] = df.index

        res = list(training_loop(df, [0, 1], "label", 4, 3, shuffle=True))
        for f, i in res:
            np.testing.assert_almost_equal(df.loc[i][[0, 1]].values, f[:, -1, :])
