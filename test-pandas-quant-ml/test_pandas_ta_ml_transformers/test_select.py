from unittest import TestCase
from config import DF_TEST, DF_TEST_MULTI, DF_TEST_MULTI_ROW_MULTI_COLUMN, DF_TEST_MULTI_ROW
from pandas_quant_ml.transformers import *
from pandas_quant_ml.patched import pd
from pandas_df_commons import monkey_patch_dataframe as patch_commons
from pandas_quant_ml.transformers.transform import Select, SelectJoin


class TestTrasformationChain(TestCase):

    def test_Select(self):
        t = Select("Open", "Close") >> PercentChange() >> LambertGaussianizer() >> Rescale()
        df = DF_TEST[:10]

        tdf = t(df)
        idf = t.inverse(tdf)

        pd.testing.assert_frame_equal(idf, df[["Open", "Close"]])

    def test_multi_Select(self):
        t = SelectJoin(
            Select("Open", "High", "Low", "Close") >> GapUpperLowerBody() >> Rescale(),
            Select("Volume") >> PercentChange() >> Rescale(),
        )

        df = DF_TEST[:10]

        tdf = t(df)
        idf = t.inverse(tdf)

        pd.testing.assert_frame_equal(idf, df[["Open", "High", "Low", "Close", "Volume"]].astype(float))

    def test_select_with_rename(self):
        t = SelectJoin(
            Select("Close", rename='LogReturns') >> PercentChange() >> LogNormalizer(),
            Select("Close", rename='Lambert') >> PercentChange() >> LambertGaussianizer(),
        )

        df = DF_TEST[:10]

        tdf = t(df)
        idf = t.inverse(tdf)

        pd.testing.assert_frame_equal(idf, df[["Close"]])
