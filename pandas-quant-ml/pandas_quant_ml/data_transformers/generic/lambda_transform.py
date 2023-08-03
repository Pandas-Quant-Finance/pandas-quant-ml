from __future__ import annotations

from typing import Callable, Iterable, Any, List

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class Lambda(DataTransformer):

    def __init__(
            self,
            func: Callable[[pd.DataFrame], pd.DataFrame],
            *args: Any,
            inv_func: Callable[[pd.DataFrame, ...], pd.DataFrame] = None,
            names: str|Iterable[str]|Callable[[str], str]=None,
            **kwargs
    ):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.inv_func = inv_func
        self.names = [names] if names is not None and not isinstance(names, (Iterable, Callable)) else names

        self._original_names = None

    def _fit(self, df: pd.DataFrame):
        self._original_names = df.columns

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.func(df, *self.args, **self.kwargs)
        if isinstance(df, pd.Series): df = df.to_frame()
        
        if self.names is not None:
            df = df.rename(columns=dict(zip(df.columns, self.names)), level=-1)

        return df.dropna()

    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return self.inv_func(df).rename(columns=self._original_names) if self.inv_func is not None else None
