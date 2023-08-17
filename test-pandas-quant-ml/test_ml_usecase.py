from unittest import TestCase

from pandas_df_commons.math import col_div
from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.filter.outlier import Winsorize
from pandas_quant_ml.data_transformers.generic.lambda_transform import Lambda
from pandas_quant_ml.data_transformers.generic.selection import Select, SelectJoin
from pandas_quant_ml.data_transformers.generic.shift import PredictDateTimePeriods
from pandas_quant_ml.data_transformers.generic.windowing import MovingWindow
from pandas_quant_ml.data_transformers.normalizer.normalized_returns import CalcNormalisedReturns
from pandas_quant_ml.data_transformers.scale.zscore import RollingZScore
from pandas_ta.technical_analysis import ta_macd
from testing_data import DF_AAPL


class TestFullMLUseCse(TestCase):

    def test_mom_trans_sample(self):
        looper = TrainTestLoop(
            # features
            Select("Close")\
                >> Winsorize(252, 5)\
                >> SelectJoin(
                    CalcNormalisedReturns([1, 21, 63, 126, 252], 60, names=["norm_daily_return", "norm_monthly_return", "norm_quarterly_return", "norm_biannual_return", "norm_annual_return"]),
                    SelectJoin(*[
                        Lambda(
                            lambda df, *args: col_div(
                                ta_macd(df, *args).droplevel(0, axis=1)['macd'],
                                df.rolling(63).std()
                            ), *args, names=[f"macd_{args[0]}_{args[1]}"]) \
                        >> RollingZScore(252)
                        for args in [(8, 24, 9), (16, 48, 9), (32, 96, 9)]])
                ) >> MovingWindow(252),
            # labels: 0.15 = 15% volatility target
            Select("Close") \
            >> Winsorize(252, 5) \
            >> CalcNormalisedReturns(1, 60, 0.15, names="target_return") \
            >> PredictDateTimePeriods(1) \
            >> MovingWindow(252),
            include_frame_name_category=True,
            # feature_shape=(252, 9)
        )

        train, test = looper.train_test_iterator([("AAPL", DF_AAPL)], train_test_split_ratio=0.9, batch_size=100,) #, nth_row_only=252)
        for t in train:
            print(t[1].shape)
        for t in test:
            print(t[1].shape)
