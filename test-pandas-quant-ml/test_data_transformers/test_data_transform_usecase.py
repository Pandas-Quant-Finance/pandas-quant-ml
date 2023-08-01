from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.math import col_div
from pandas_quant_ml.data_transformers.filter.outlier import Winsorize
from pandas_quant_ml.data_transformers.generic.lambda_transform import Lambda
from pandas_quant_ml.data_transformers.generic.selection import Select, SelectJoin
from pandas_quant_ml.data_transformers.normalizer.normalized_returns import CalcNormalisedReturns
from pandas_quant_ml.data_transformers.scale.zscore import RollingZScore
from pandas_ta.technical_analysis import ta_macd
from tesing_data import DF_AAPL


class TestDataTransformerUseCase(TestCase):

    def test_data_transformation_mom_trans(self):
        def legacy(df_asset: pd.DataFrame, price='Close') -> pd.DataFrame:
            df_asset[df_asset[price] <= 1e-8] = np.nan

            # winsorize using rolling 5X standard deviations to remove outliers
            df_asset["srs"] = df_asset[price]
            ewm = df_asset["srs"].ewm(halflife=252)
            means = ewm.mean()
            stds = ewm.std()
            df_asset["srs"] = np.minimum(df_asset["srs"], means + 5 * stds)
            df_asset["srs"] = np.maximum(df_asset["srs"], means - 5 * stds)

            df_asset["daily_returns"] = (df_asset["srs"]).pct_change()
            df_asset["daily_vol"] = df_asset["daily_returns"].ewm(span=60, min_periods=60).std()

            # calculates volatility scaled returns for annualised (np.sqrt(252)) VOL_TARGET of 15%
            df_asset["target_returns"] = df_asset["daily_returns"] * 0.15 / (
                        df_asset["daily_vol"] * np.sqrt(252)).shift(1)

            # and shift to be next day returns
            df_asset["target_returns"] = df_asset["target_returns"].shift(-1)

            # normalize returns (z-score)
            def calc_normalised_returns(day_offset):
                return (
                        df_asset["srs"].pct_change(day_offset)
                        / df_asset["daily_vol"]  # interesting we dived the returns by the standard deviation
                        / np.sqrt(day_offset)  # and multiply it back to the number of days
                )

            df_asset["norm_daily_return"] = calc_normalised_returns(1)
            df_asset["norm_monthly_return"] = calc_normalised_returns(21)
            df_asset["norm_quarterly_return"] = calc_normalised_returns(63)
            df_asset["norm_biannual_return"] = calc_normalised_returns(126)
            df_asset["norm_annual_return"] = calc_normalised_returns(252)

            trend_combinations = [(8, 24), (16, 48), (32, 96)]
            for short_window, long_window in trend_combinations:
                macd = ta_macd(df_asset["srs"].dropna(), short_window, long_window, 9).iloc[:, 0]
                q = macd / df_asset["srs"].rolling(63).std()
                df_asset[f"macd_{short_window}_{long_window}"] = q / q.rolling(252).std()

            return df_asset

        pipeline = Select("Close") >> Winsorize(252, 5)\
            >> SelectJoin(
                CalcNormalisedReturns([1, 21, 63, 126, 252], 60, names=["norm_daily_return", "norm_monthly_return", "norm_quarterly_return", "norm_biannual_return", "norm_annual_return"]),
                SelectJoin(*[
                    Lambda(
                        lambda df, *args: col_div(
                            ta_macd(df, *args).droplevel(0, axis=1)['macd'],
                            df.rolling(63).std()
                        ), *args, names=[f"macd_{args[0]}_{args[1]}"]) \
                    >> RollingZScore(252, demean=False)
                    for args in [(8, 24, 9), (16, 48, 9), (32, 96, 9)]])
            ) # >> TODO Window(252, 252)

            # TODO     df_asset["target_returns"] = df_asset["daily_returns"] * VOL_TARGET / (df_asset["daily_vol"] * np.sqrt(252)).shift(1)

        transformed, _ = pipeline.fit_transform(DF_AAPL[["Open", "Close"]], 1)
        print(transformed.tail().columns)
        print(legacy(DF_AAPL).tail().columns)
        np.testing.assert_array_almost_equal(
            legacy(DF_AAPL).tail().drop(["Open", "High", "Low", "Close", "Volume", "srs", 'Dividends', 'Stock Splits', 'daily_returns', 'daily_vol', 'target_returns'], axis=1),
            transformed.tail()
        )
        print(transformed.tail().columns)
        print(legacy(DF_AAPL).tail().columns)
        res = []
        pipeline.transform(DF_AAPL, res)
        print(len(res))
        print(transformed.max().max())

