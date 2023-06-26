from unittest import TestCase
from config import DF_TEST, DF_TEST_MULTI, DF_TEST_MULTI_ROW_MULTI_COLUMN, DF_TEST_MULTI_ROW
from pandas_quant_ml.transformers import *
from pandas_quant_ml.patched import pd
from pandas_df_commons import monkey_patch_dataframe as patch_commons
from pandas_quant_ml.transformers.transform import Select, SelectJoin


class TestTransformationChain(TestCase):

    def test_example_usecase(self):
        features, labels, label_inverter = ml_features_labels(
            DF_TEST_MULTI,
            SelectJoin(
                Select("Open", "High", "Low", "Close") >> GapUpperLowerBody(),
                Select("Volume") >> PercentChange() >> LogNormalizer(),
                Select("Close") >> ZScore(20, exponential=True),
            ) >> ShiftAppend(30),
            Select("Close") >> PercentChange() >> CumProd(5),
            labels_shift=-5
        )

        # label_inverter(labels)
        print(features.tail())
        print(labels.tail())

    def test_usecase_mom_trans(self):
        """
        ("ticker", DataTypes.CATEGORICAL, InputTypes.ID),
            ("date", DataTypes.DATE, InputTypes.TIME),
            ("target_returns", DataTypes.REAL_VALUED, InputTypes.TARGET),


        VOL_THRESHOLD = 5  # multiple to winsorise by
        VOL_LOOKBACK = 60  # for ex-ante volatility
        VOL_TARGET = 0.15  # 15% volatility target
        HALFLIFE_WINSORISE = 252

        # winsorize using rolling 5X standard deviations to remove outliers
        df_asset["srs"] = df_asset[price]
        ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
        means = ewm.mean()
        stds = ewm.std()
        df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
        df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

        df_asset["daily_returns"] = ta_returns(df_asset["srs"])  # srs is outliers removed
        df_asset["daily_vol"] = df_asset["daily_returns"].ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()

        # calculates volatility scaled returns for annualised (np.sqrt(252)) VOL_TARGET of 15%
        df_asset["target_returns"] = df_asset["daily_returns"] * VOL_TARGET / (df_asset["daily_vol"] * np.sqrt(252)).shift(1)

        # and shift to be next day returns
        df_asset["target_returns"] = df_asset["target_returns"].shift(-1)

        # normalize returns (z-score)
        def calc_normalised_returns(day_offset):
            return (
                    ta_returns(df_asset["srs"], day_offset).iloc[:, 0]
                    / df_asset["daily_vol"]  # interesting we dived the returns by the standard deviation
                    / np.sqrt(day_offset)   # and multiply it back to the number of days
            )

        df_asset["norm_daily_return"] = calc_normalised_returns(1)
        df_asset["norm_monthly_return"] = calc_normalised_returns(21)
        df_asset["norm_quarterly_return"] = calc_normalised_returns(63)
        df_asset["norm_biannual_return"] = calc_normalised_returns(126)
        df_asset["norm_annual_return"] = calc_normalised_returns(252)

        trend_combinations = [(8, 24), (16, 48), (32, 96)]
        for short_window, long_window in trend_combinations:
            # calculate stanrad macd and the mormalize by standard deviation
            macd = ta_macd(df_asset["srs"].dropna(), short_window, long_window, 9)['macd']
            q = macd / df_asset["srs"].rolling(63).std()
            df_asset[f"macd_{short_window}_{long_window}"] = q / q.rolling(252).std()

        """
        # TODO we need to separate fit and transform just like the sklearn API
        features, labels, label_inverter = ml_features_labels(
            DF_TEST_MULTI,  # how to encorporate ta functions, already here at frame level?
            SelectJoin(
                # ("ticker", DataTypes.CATEGORICAL, InputTypes.ID) => sklearn.preprocessing.LabelEncoder().fit(srs.values)
                # Select("Close") >> Winsorize() >> SelectJoin(
                    # CalcNormalisedReturns(1, "norm_daily_return"),
                    # CalcNormalisedReturns(21, "norm_monthly_return"),
                    # CalcNormalisedReturns(63, "norm_quarterly_return"),
                    # CalcNormalisedReturns(126, "norm_biannual_return"),
                    # CalcNormalisedReturns(252, "norm_annual_return"),
                    # Lambda(lambda df: df.ta.macd(...), "macd_8_24"),
                    # Lambda(lambda df: df.ta.macd(...), "macd_16_48"),
                    # Lambda(lambda df: df.ta.macd(...), "macd_32_96"),
                # )
            ) >> ShiftAppend(30),
            Select("Close") >> PercentChange(),
            labels_shift=-1
        )
