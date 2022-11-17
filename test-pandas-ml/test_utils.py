from unittest import TestCase

from pandas_ta_ml.utils.rescaler import ReScaler


class TestUtils(TestCase):

    def test_rescaling_rows(self):
        self.assertEquals(ReScaler((10, 8), (1, -1))(10), 1)
        self.assertEquals(ReScaler((10, 8), (1, -1))(8), -1)
        self.assertEquals(ReScaler((10, 8), (1, -1))(9), 0)

        self.assertEquals(ReScaler((8, 10), (1, -1))(10), -1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(8), 1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(9), 0)

        self.assertEquals(ReScaler((8, 10), (1, -1))(10), -1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(8), 1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(9), 0)
