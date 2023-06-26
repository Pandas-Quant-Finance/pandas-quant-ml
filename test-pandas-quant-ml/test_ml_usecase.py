from unittest import TestCase

import pandas as pd


class TestFullMLUseCse(TestCase):

    def test_mom_trans_sample(self):
        print("\n")
        def frames(files):
            for file in files:
                print("yield", file)
                yield (file, pd.DataFrame({}))

        print("make iter")
        x = iter(frames(["a", "b", "c"]))

        print("start loop")
        for i in x:
            print("lala")

        """
        object(
            frames=iter(frames(["a", "b", "c"]))
            features=SelectJoin(
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
            labels=Select("Close") >> PercentChange() >> Shift(-1),
            train_test_split_ratio=0.75,
            batch_size=128,
            look_back_window=None,
            shuffle=False,
            epochs=2,            
        )
        """


        pass