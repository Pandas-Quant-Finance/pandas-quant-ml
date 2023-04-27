from unittest import TestCase

import pandas as pd
import numpy as  np
from pandas_quant_ml.data_generators.ml_traning_loop import BatchGenerator

DF = pd.DataFrame(np.random.random((30, 2)), index=pd.MultiIndex.from_product([["A", "B"], range(15)]))


class TestMlBatching(TestCase):

    def test_Batch_index_iterator(self):
        batches = BatchGenerator(DF, 4, )
        for i, b in enumerate(batches):
            pd.testing.assert_frame_equal(
                pd.DataFrame(b[1]),
                pd.DataFrame(batches[i][1]),
            )

            pd.testing.assert_index_equal(b[0], batches[i][0])

    def test_batch_simple(self):
        batches = [b for _, b in BatchGenerator(DF, 4, )]
        res = np.vstack(batches)
        print(res)

        self.assertGreater(len(batches), 2)
        self.assertLess(len(batches[-1]), 4)
        np.testing.assert_almost_equal(DF.values, res, 5)

    def test_batch_window_simple(self):
        batches = [b for _, b in BatchGenerator(DF, 4, 3)]

        self.assertGreater(len(batches), 2)
        self.assertLess(len(batches[-1]), 4)
        np.testing.assert_almost_equal(DF[:3].values, batches[0][0], 5)
        np.testing.assert_almost_equal(DF[-3:].values, batches[-1][-1], 5)

    def test_batch_shuffled(self):
        batches1 = [b for _, b in BatchGenerator(DF, 4, shuffle=False)]
        batches2 = [b for _, b in BatchGenerator(DF, 4, shuffle=True)]
        self.assertEqual(len(batches1), len(batches2))

    def test_batch_window_shuffled(self):
        batches1 = [b for _, b in BatchGenerator(DF, 4, 3, shuffle=False)]
        batches2 = [b for _, b in BatchGenerator(DF, 4, 3, shuffle=True)]

        self.assertEqual(len(batches1), len(batches2))
        for i in range(len(batches1)):
            self.assertEqual(batches1[i].shape, batches2[i].shape, str(i))
            self.assertFalse(np.array_equal(batches1[i], batches2[i]), str(i))

