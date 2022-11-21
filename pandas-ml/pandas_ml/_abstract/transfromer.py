from abc import abstractmethod

import pandas as pd


class Transformer(object):

    def __init__(self, *args, **kwargs):
        self.df: pd.DataFrame = None

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        self.df = df
        return self.transform(df)

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def inverse(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass
