from __future__ import annotations

from copy import deepcopy
from functools import partial
from types import MethodType
from typing import List, Callable, Tuple

import pandas as pd

from pandas_ta.pandas_ta_utils.decorators import for_each_top_level_row_aggregate, for_each_top_level_column_aggregate
from pandas_ta_ml._abstract.transfromer import Transformer
from pandas_df_commons.indexing import get_columns


def ml_transform(df: pd.DataFrame, *transformers: Flow, drop_untransformed=True, return_transformers=False, patch_df_4_inverse=True):
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
        def agg(results):
            return (
                pd.concat([v[0] for v in results.values()], keys=results.keys(), axis=axis),
                pd.concat([pd.DataFrame({"Transformer": [v[1]]}) for v in results.values()], keys=results.keys(), axis=axis)
            )
        return agg

    @for_each_top_level_row_aggregate(aggregator(0))
    @for_each_top_level_column_aggregate(aggregator(1))
    def transform(_df):
        flows = deepcopy(Flows(*transformers, drop_untransformed=drop_untransformed))
        return flows.transform(_df), flows

    def inverse_flow_method(self, flows):
        if isinstance(flows, Transformer):
            return flows.inverse(self)
        else:
            # TODO for each row/column we need a different invertor
            #flows.loc[row, column].inverse(self.loc[row or column])
            pass

    df, fitted_flows = transform(df)

    if patch_df_4_inverse:
        df.inverse = MethodType(partial(inverse_flow_method, flows=fitted_flows), df)

    # return transformed data with or without the fitted transformer
    return (df, fitted_flows) if return_transformers else df


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

