from unittest import TestCase
from config import DF_TEST, DF_TEST_MULTI, DF_TEST_MULTI_ROW_MULTI_COLUMN
from pandas_ml.transformers import *
from pandas_ml.patched import pd
from pandas_df_commons import monkey_patch_dataframe as patch_commons


class TestTrasformationChain(TestCase):

    def test_invertible_chain(self):
        df = DF_TEST.ml.transform(
            Flow("Open", "Close") >> PercentChange() >> LambertGaussianizer() >> Rescale(),
            Flow("Volume") >> PercentChange() >> Rescale(),
        )

        self.assertTrue(hasattr(df, "inverse"))
        df = df.inverse()
        #print(df)
        pd.testing.assert_frame_equal(df, DF_TEST[["Open", "Close", "Volume"]].astype(float))

    def test_invertible_chain_multiindex(self):
        patch_commons()
        pd.testing.assert_frame_equal(
            DF_TEST_MULTI.ml.transform(
                Flows(
                    Flow("Open", "Close") >> PercentChange() >> Rescale(),
                    Flow("Volume") >> PercentChange() >> Rescale(),
                )
            ).inverse().sort_index(axis=1),
            DF_TEST_MULTI.X[["Open", "Close", "Volume"]].sort_index(axis=1).astype(float)
        )

    def test_chain_keep_untransformed(self):
        df = DF_TEST.ml.transform(
            Flows(
                Flow("Volume") >> PercentChange() >> Rescale(),
                drop_untransformed=False,
            )
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

        self.assertListEqual(df.columns.tolist(), ['gap', 'body', 'shadow', 'position', 'Volume'])
        pd.testing.assert_frame_equal(df.inverse(), df)

    def test_example_usecase(self):
        features, labels, label_inverter = DF_TEST_MULTI.ml.features_labels(
            Flows(
                Flow("Close") >> PercentChange() >> ShiftAppend(20)
            ),
            Flows(
                Flow("Close") >> PercentChange() >> CumProd(5)  # shift -5
            )
        )

        label_inverter(labels)
        print("")