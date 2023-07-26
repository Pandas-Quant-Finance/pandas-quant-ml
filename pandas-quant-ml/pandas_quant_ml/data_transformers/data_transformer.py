from __future__ import annotations

from typing import Callable, List, Tuple

import pandas as pd
from abc import ABC, abstractmethod


class DataTransformer(ABC):

    def __init__(self):
        self._previous: DataTransformer = None
        self._next: DataTransformer = None
        self._is_fitted = False

    def __rshift__(self, other: 'DataTransformer'):
        self._next = other
        other._previous = self
        return other

    def fit_transform(self, df: pd.DataFrame, reserved_data_length: int, queue: bool = False) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        assert len(df) - reserved_data_length > 0, f"need data for training len={len(df)} reserved for test={reserved_data_length}"
        self.fit(df.iloc[:max(len(df) - reserved_data_length, -1)])
        return self.transform(df, queue)

    def fit(self, df: pd.DataFrame):
        assert not self._is_fitted, f"{self.__class__} is already fitted"
        if isinstance(df, pd.Series): df = df.to_frame()

        if self._previous is not None:
            self._previous.fit(df)
            df, _ = self._previous.transform(df)

        self._fit(df)
        self._is_fitted = True

    def transform(self, df: pd.DataFrame, queue: List[pd.DataFrame] | bool = None) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        assert self._is_fitted, f"{self.__class__} need to be fitted first"
        if isinstance(df, pd.Series): df = df.to_frame()
        if queue is None or queue is True: queue = [df]

        if self._previous is not None:
            df, _ = self._previous.transform(df, queue)

        df = self._transform(df)
        if isinstance(queue, List): queue.append(df)

        return df, queue[:-1] if isinstance(queue, List) else queue

    def inverse(self, df: pd.DataFrame, queue: List[pd.DataFrame]):
        assert self._is_fitted, f"{self.__class__} need to be fitted first"
        if isinstance(df, pd.Series): df = df.to_frame()

        df = self._inverse(df, queue.pop())
        if self._previous is not None:
            df = self._previous.inverse(df, queue)

        return df

    def reset(self):
        if self._previous is not None:
            self._previous.reset()

        self._is_fitted = False

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError("Inverse not implemented for " + self.__class__.__name__)

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Inverse not implemented for " + self.__class__.__name__)

    def _inverse(self, df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Inverse not implemented for " + self.__class__.__name__)


