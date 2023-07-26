from __future__ import annotations

from typing import Callable, Iterable, List

import pandas as pd

from pandas_quant_ml.data_transformers.data_transformer import DataTransformer


class Select(DataTransformer):

    def __init__(self, *columns, names: Iterable[str]|Callable[[str], str]=None,):
        super().__init__()
        self.columns = list(columns)
        self.names = names

        self._original_names = None

    def _fit(self, df: pd.DataFrame):
        self._original_names = self.columns

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self.columns]
        if self.names is not None:
            df = df.rename(
                columns=self.names if callable(self.names) else dict(zip(df.columns, self.names))
            )
        return df

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=dict(zip(df.columns, self._original_names)))


class SelectJoin(DataTransformer):

    def __init__(self, *selectors: DataTransformer):
        super().__init__()
        self.selectors = list(selectors)
        self.resulting_columns = None

    def transform(self, df: pd.DataFrame, queue: List[pd.DataFrame] = None):
        if isinstance(df, pd.Series): df = df.to_frame()
        if queue is None or queue is True: queue = [df]

        if self._previous is not None:
            df, _ = self._previous.transform(df, queue)

        queues = [[] for _ in self.selectors]
        dfs = [se.transform(df, q)[0] for se, q in zip(self.selectors, queues)]
        self.resulting_columns = [f.columns.tolist() for f in dfs]
        if isinstance(queue, List): queue.append(queues)

        return pd.concat(dfs, axis=1, join='inner', sort=True), queue

    def reset(self):
        super().reset()
        for s in self.selectors:
            s.reset()

    def _fit(self, df: pd.DataFrame):
        for s in self.selectors:
            s.fit(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # should never get here
        pass


