from unittest import TestCase

from pandas_quant_ml.data_transformers.filter.outlier import Winsorize
from pandas_quant_ml.data_transformers.generic.selection import Select, SelectJoin
from pandas_quant_ml.data_transformers.normalizer.normalized_returns import CalcNormalisedReturns
from testing_data import DF_AAPL


class TestDataTransformer(TestCase):

    def test_simple(self):
        dt = Select("Close")\
             >> Winsorize(252, 5)\
             >> CalcNormalisedReturns([1, 21, 63, 126, 252], 60,)

        dt.fit_transform(DF_AAPL, 20)
        print(dt.transform(DF_AAPL))
        #FIXME
        try:
            print(dt.inverse(dt.transform(DF_AAPL)))
        except Exception as e:
            print(e)

    def test_join(self):
        dt = Select("Close")\
             >> Winsorize(252, 5)\
             >> SelectJoin(
                CalcNormalisedReturns([1, 21, 63, 126, 252], 60, )
             )

        dt.fit_transform(DF_AAPL, 20)
        visitor = []
        print(dt.transform(DF_AAPL, visitor))
        #FIXME
        try:
            print(dt.inverse(dt.transform(DF_AAPL)))
        except Exception as e:
            print(e)

