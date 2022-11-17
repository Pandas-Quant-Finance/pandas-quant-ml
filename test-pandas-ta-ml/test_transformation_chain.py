from unittest import TestCase
from config import DF_TEST
from pandas_ta_ml.transformers import *
from pandas_ta_ml.patched import pd


class TestTrasformationChain(TestCase):

    def test_chain(self):
        df = DF_TEST.ml.transform(
            Flow("Open", "High", "Low", "Close") >> LogNormalizer() >> Rescale(),
            Flow("Volume") >> (lambda s: s.pct_change())
        )

        print(df)

    def test_invertible_chain(self):
        df = DF_TEST.ml.transform(
            Flow("Open", "Close") >> PercentChange() >> LambertGaussianizer() >> Rescale(),
            Flow("Volume") >> PercentChange() >> Rescale(),
        )

        self.assertTrue(hasattr(df, "inverse"))
        df = df.inverse()
        #print(df)
        pd.testing.assert_frame_equal(df, DF_TEST[["Open", "Close", "Volume"]].astype(float))

    def test_chain_keep_untransformed(self):
        df = DF_TEST.ml.transform(
            Flow("Volume") >> PercentChange() >> Rescale(),
            drop_untransformed=False,
        )

        df = df.inverse()
        # Note that we can not guarantee the order
        pd.testing.assert_frame_equal(df[DF_TEST.columns.tolist()], DF_TEST.astype(float))

    def test_goal(self):
        src = DF_TEST[["Open", "High", "Low", "Close", "Volume"]]

        df = src.ml.transform(
            Flow("Open", "High", "Low", "Close") >> PositionalBar() >> Flows(
                Flow("gap") >> LambertGaussianizer() >> Rescale(),
                Flow("body") >> LambertGaussianizer() >> Rescale(),
                Flow("shadow") >> LambertGaussianizer() >> Rescale(),
                Flow("position") >> LambertGaussianizer() >> Rescale(),
            ),
            Flow("Volume") >> PercentChange() >> LogNormalizer()
        )

        pd.testing.assert_frame_equal(df.inverse(), df)