from __future__ import annotations

from copy import deepcopy
from functools import partial
from types import MethodType
from typing import List, Callable, Tuple

import pandas as pd

from pandas_df_commons.indexing.decorators import foreach_top_level_row_aggregate, foreach_top_level_column_aggregate
from pandas_ml._abstract.transfromer import Transformer
from pandas_df_commons.indexing import get_columns


# note top level rows and columns will be handled inside the function
def ml_features_labels(df: pd.DataFrame, feature_transformers: Flows, label_transformers: Flows, ):
    dff = ml_transform(df, feature_transformers, return_transformers=False, patch_df_4_inverse=False)
    dfl, lt = ml_transform(df, label_transformers, return_transformers=True, patch_df_4_inverse=False)

    return dff, dfl, Inverse(lt)


# note top level rows and columns will be handled inside the function
def ml_transform(df: pd.DataFrame, transformers: Flows, return_transformers=False, patch_df_4_inverse=True):
    """
    All transformers return a tuple of dataframe and an inverse function.
    Therefore, (in contrast to analytics functions) none of the transformers is decorated and each frame gets treated as
    a whole regardless if logically repeating data is present at the top level MultiIndex (like bar data of multiple
    stocks). In order to take such logically repeating datasets into account this function has to be used as the
    entrypoint - i.e.:
        df.ml.transform(
            Col("Open", "High", "Low", "Close") >> (lambda df: df.ml.positional_bar().ml.transform(
                Col("gap") >> (lambda df: df.ml.lambert_gaussianizer()),
                Col("body") >> (lambda df: df.ml.lambert_gaussianizer()),
                Col("shadow") >> (lambda df: df.ml.lambert_gaussianizer()),
                Col("position") >> (lambda df: df.ml.lambert_gaussianizer()),
            )),
            Col("Volume") >> (lambda s: s.pct_change())
        )
    """

    def aggregator(axis):
        def agg(results, level):
            return (
                pd.concat([v[0] for v in results.values()], keys=results.keys(), axis=axis),
                pd.concat([pd.DataFrame({"Transformer": [v[1]]}) for v in results.values()], keys=results.keys(), axis=axis)
            )
        return agg

    @foreach_top_level_row_aggregate(aggregator(0))
    @foreach_top_level_column_aggregate(aggregator(1))
    def transform(_df):
        flows = deepcopy(transformers)
        return flows.transform(_df), flows

    df, fitted_flows = transform(df)

    if patch_df_4_inverse:
        df.inverse = MethodType(Inverse(fitted_flows), df)

    # return transformed data with or without the fitted transformer
    return (df, fitted_flows) if return_transformers else df


class Inverse(object):

    def __init__(self, transformer: Transformer | pd.DataFrame):
        super().__init__()
        self.transformer = transformer

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        if isinstance(self.transformer, pd.DataFrame):
            tfs = self.transformer
            union_frames = []

            for ri, row in tfs.iterrows():
                join_frames = []
                for ci, value in row.items():
                    t = tfs.loc[ri, ci]
                    if isinstance(ri, tuple) and isinstance(ci, tuple):
                        join_frames.append(t.inverse(df.loc[ri[0], ci[0]]))
                    if isinstance(ri, tuple):
                        join_frames.append(t.inverse(df.loc[ri[0]]))
                    if isinstance(ci, tuple):
                        join_frames.append(t.inverse(df[ci[0]]))
                    else:
                        raise ValueError("row or column has to be multiindex")

                union_frames.append(
                    pd.concat(
                        join_frames,
                        keys=tfs.columns.get_level_values(0) if isinstance(tfs.columns, pd.MultiIndex) else None,
                        axis=1,
                    )
                )

            return pd.concat(
                union_frames,
                keys=tfs.index.get_level_values(0) if isinstance(tfs.index, pd.MultiIndex) else None,
                axis=0
            )
        else:
            return self.transformer.inverse(df, **kwargs)


class Flows(Transformer):

    def __init__(self, *flows: Flow, drop_untransformed=True):
        super().__init__()
        self.flows: Tuple[Flow] = flows
        self.drop_untransformed = drop_untransformed
        self._selected_columns = None
        self._inverse = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = [f(df) for f in self.flows]
        _df = pd.concat([frame for frame, _ in transformed], axis=1)

        def inverse_flow(frame):
            frame = pd.concat([iflow(frame)[0] for _, iflow in transformed], axis=1)
            return frame

        self._inverse = inverse_flow

        if not self.drop_untransformed:
            selected_columns = []
            for flow in self.flows:
                for c in flow._selected_columns:
                    if c in df.columns:
                        selected_columns.append(c)

            self._selected_columns = selected_columns
            _df = _df.join(df.drop(self._selected_columns, axis=1))

        return _df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.drop_untransformed:
            return self._inverse(df).join(df.drop(self._selected_columns, axis=1))
        else:
            return self._inverse(df)


class Flow(object):

    def __init__(self, *columns) -> None:
        self.columns = list(columns)
        self.transformers: List[Transformer] = []
        self._selected_columns = None

    def __rshift__(self, other: Transformer | Callable[[pd.DataFrame], pd.DataFrame]):
        self.transformers.append(other)
        return self

    def __call__(self, df, *args, **kwargs) -> Tuple[pd.DataFrame, Flow | None]:
        # subselect columns
        df = get_columns(df, self.columns)
        if df.ndim < 2: df = df.to_frame()
        self._selected_columns = df.columns

        # transform all data
        for t in self.transformers:
            df = t(df)

        # assemble inverter functions
        inverter = Flow(*self.columns)
        for t in reversed(self.transformers):
            if not hasattr(t, 'inverse'):
                return df, None

            inverter = inverter >> t.inverse

        # return transformed dataframe and inverter functions
        return df, inverter

