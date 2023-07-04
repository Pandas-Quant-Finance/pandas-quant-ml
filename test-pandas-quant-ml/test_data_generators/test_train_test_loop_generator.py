from unittest import TestCase

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.filter.outlier import Winsorize
from pandas_quant_ml.data_transformers.generic.selection import Select
from tesing_data import DF_AAPL


class TestTrainTestLoopGenerator(TestCase):

    def test_train_test_iterator(self):
        ttl = TrainTestLoop(
            Select("Close") >> Winsorize(252, 5),
            Select("Close") >> Winsorize(252, 5),
            batch_size=100
        )

        train, test = ttl.train_test_iterator(DF_AAPL)

        self.assertListEqual([t[0].shape[0] for t in train], [100, 100, 100, 100, 100, 13])
        self.assertListEqual([t[0].shape[0] for t in train.to_repeating_iterator(2)], [100, 100, 100, 100, 100, 13] * 2)
        self.assertListEqual([t[0].shape[0] for t in test.to_repeating_iterator(2)], [100, 70] * 2)

