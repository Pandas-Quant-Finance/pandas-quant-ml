from unittest import TestCase

from matplotlib.figure import Figure

from pandas_quant_ml.patched import pd
from pandas_quant_ml.transformers import Select


class TestPatchedDataFrame(TestCase):

    def test_analytics_extension(self):
        df = pd.DataFrame({"Close": range(10)})
        fig = df.ml.qqplot(return_fig=True)

        self.assertIsInstance(fig, Figure)

    def test_transformer_extension(self):
        df = pd.DataFrame({"Close": range(10)})
        t = df.ml.transform(Select("Close"))

        pd.testing.assert_frame_equal(df, t)
