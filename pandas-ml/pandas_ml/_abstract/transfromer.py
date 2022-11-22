from abc import abstractmethod

import pandas as pd


class Transformer(object):

    def __init__(self, **kwargs):
        self.df: pd.DataFrame = None
        self.left: Transformer = None
        self.kwargs = kwargs

    def __rshift__(self, other: 'Transformer'):
        other.left = self
        return other

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.left is not None:
            df = self.left(df)

        df = self._transform(df)
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._inverse(df)

        if self.left is not None:
            df = self.left.inverse(df)

        return df

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
