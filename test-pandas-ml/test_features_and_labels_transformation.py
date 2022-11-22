from unittest import TestCase
from config import DF_TEST, DF_TEST_MULTI, DF_TEST_MULTI_ROW_MULTI_COLUMN, DF_TEST_MULTI_ROW
from pandas_ml.transformers import *
from pandas_ml.patched import pd
from pandas_df_commons import monkey_patch_dataframe as patch_commons
from pandas_ml.transformers.transform import Select, SelectJoin


class TestTransformationChain(TestCase):

    def test_example_usecase(self):
        features, labels, label_inverter = ml_features_labels(
            DF_TEST_MULTI,
            Select("Open", "High", "Low", "Close") >> GapUpperLowerBody(),
            Select("Close") >> PercentChange() >> CumProd(5),
            labels_shift=-5
        )

        # label_inverter(labels)
        print(features.tail())
        print(labels.tail())