from unittest import TestCase

from config import DF_TEST, DF_TEST_MULTI, DF_TEST_MULTI_ROW_MULTI_COLUMN, DF_TEST_MULTI_ROW
from pandas_df_commons import monkey_patch_dataframe as patch_commons
from pandas_quant_ml.patched import pd
from pandas_quant_ml.transformers import *
from pandas_quant_ml.transformers.transform import Select, SelectJoin


class TestTransformationChain(TestCase):

    def test_invertible_chain(self):
        df, inv = ml_transform(
            DF_TEST,
            SelectJoin(
                Select("Open", "Close") >> PercentChange() >> LambertGaussianizer() >> Rescale(),
                Select("Volume") >> PercentChange() >> Rescale(),
            ),
            return_inverter=True
        )

        pd.testing.assert_frame_equal(inv(df), DF_TEST[["Open", "Close", "Volume"]].astype(float))

    def test_invertible_chain_multiindex_row(self):
        patch_commons()
        df, inv = ml_transform(
            DF_TEST_MULTI_ROW,
            SelectJoin(
                Select("Open", "Close") >> PercentChange() >> Rescale(),
                Select("Volume") >> PercentChange() >> Rescale(),
            ),
            return_inverter=True
        )

        pd.testing.assert_frame_equal(
            inv(df), DF_TEST_MULTI_ROW[["Open", "Close", "Volume"]].astype(float)
        )

    def test_invertible_chain_multiindex_col(self):
        patch_commons()
        df, inv = ml_transform(
            DF_TEST_MULTI,
            SelectJoin(
                Select("Open", "Close") >> PercentChange() >> Rescale(),
                Select("Volume") >> PercentChange() >> Rescale(),
            ),
            return_inverter=True
        )

        pd.testing.assert_frame_equal(
            inv(df).sort_index(axis=1),
            DF_TEST_MULTI.X[["Open", "Close", "Volume"]].sort_index(axis=1).astype(float)
        )

    def test_invertible_chain_multiindex_all(self):
        patch_commons()
        df, inv = ml_transform(
            DF_TEST_MULTI_ROW_MULTI_COLUMN,
            SelectJoin(
                Select("Open", "Close") >> PercentChange() >> Rescale(),
                Select("Volume") >> PercentChange() >> Rescale(),
            ),
            return_inverter=True
        )

        pd.testing.assert_frame_equal(
            inv(df).sort_index(axis=1),
            DF_TEST_MULTI_ROW_MULTI_COLUMN.X[["Open", "Close", "Volume"]].sort_index(axis=1).astype(float)
        )

    def test_goal(self):
        src = DF_TEST[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        df, inv = ml_transform(
            src,
            SelectJoin(
                Select("Open", "High", "Low", "Close") >> GapUpperLowerBody() >> SelectJoin(
                    Select("gap") >> LambertGaussianizer() >> Rescale(),
                    Select("upper") >> LambertGaussianizer() >> Rescale(),
                    Select("lower") >> LambertGaussianizer() >> Rescale(),
                    Select("body") >> LambertGaussianizer() >> Rescale(),
                ),
                Select("Volume") >> PercentChange() >> LogNormalizer()
            ),
            return_inverter=True
        )

        self.assertListEqual(df.columns.tolist(), ['gap', 'upper', 'lower', 'body', 'Volume'])
        pd.testing.assert_frame_equal(inv(df), src)
