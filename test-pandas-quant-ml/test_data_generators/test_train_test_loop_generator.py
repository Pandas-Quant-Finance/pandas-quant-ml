import shutil
from unittest import TestCase

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.filter.outlier import Winsorize
from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.utils.obj_file_cache import CACHE_ROOT_DIR
from testing_data import DF_AAPL


class TestTrainTestLoopGenerator(TestCase):

    def test_train_test_iterator(self):
        ttl = TrainTestLoop(
            Select("Close") >> Winsorize(252, 5),
            Select("Close") >> Winsorize(252, 5),
        )

        train, test = ttl.train_test_iterator(DF_AAPL, batch_size=100)

        self.assertListEqual([t[0].shape[0] for t in train], [100, 100, 100, 100, 100, 13])
        self.assertListEqual([t[0].shape[0] for t in train.to_repeating_iterator(2)], [100, 100, 100, 100, 100, 13] * 2)
        self.assertListEqual([t[0].shape[0] for t in test.to_repeating_iterator(2)], [100, 70] * 2)

    def test_train_val_test_iterator(self):
        ttl = TrainTestLoop(
            Select("Close") >> Winsorize(252, 5),
            Select("Close") >> Winsorize(252, 5),
        )

        train, val, test = ttl.train_test_iterator(DF_AAPL, train_test_split_ratio=(0.7, 0.7), batch_size=100)

        self.assertListEqual([t[0].shape[0] for t in train], [100, 100, 100, 100, 79])
        self.assertListEqual([t[0].shape[0] for t in val], [100, 44])
        self.assertListEqual([t[0].shape[0] for t in test], [61])

    def test_caching_loop_to_disk(self):
        ttl = TrainTestLoop(
            Select("Close") >> Winsorize(252, 5),
            Select("Close") >> Winsorize(252, 5),
        )

        shutil.rmtree(CACHE_ROOT_DIR.joinpath('test-3318623619066792576'), ignore_errors=True)
        for i in range(2):
            with self.assertLogs() as captured:
                train, val, test = ttl.train_test_iterator(DF_AAPL, train_test_split_ratio=(0.7, 0.7), batch_size=100, cache_key='test-3318623619066792576')
                self.assertListEqual([t[0].shape[0] for t in train], [100, 100, 100, 100, 79])
                self.assertListEqual([t[0].shape[0] for t in val], [100, 44])
                self.assertListEqual([t[0].shape[0] for t in test], [61])

                if i > 0:
                    self.assertIn("fetch cached /tmp/test-3318623619066792576", ";".join(captured.output))

        for _, inf in ttl.inference_generator(DF_AAPL, lambda x: x[:,0]):
            print(len(DF_AAPL), len(inf))
            self.assertEquals(len(DF_AAPL) - 1, len(inf))