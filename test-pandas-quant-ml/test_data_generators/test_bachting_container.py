from unittest import TestCase

import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import BatchingContainer
from testing_data.data import get_x_or


class TestBatchingContainer(TestCase):

    def test_change_batchsize(self):
        df = get_x_or()

        pd.testing.assert_frame_equal(df, pd.concat([b for b in BatchingContainer(df, None)], axis=0))
        pd.testing.assert_frame_equal(df, pd.concat([b for b in BatchingContainer(df, 12)], axis=0))

        bc = BatchingContainer(df, 13)
        self.assertEquals(len(next(iter(bc))), 13)

        pd.testing.assert_frame_equal(df, pd.concat([b for b in bc.with_batch_size(11)], axis=0))
        self.assertEquals(len(next(iter(bc.with_batch_size(11)))), 11)
