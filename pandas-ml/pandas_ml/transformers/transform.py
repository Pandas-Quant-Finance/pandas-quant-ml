from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from pandas_df_commons.indexing import get_columns
from pandas_df_commons.indexing.decorators import foreach_top_level_row_and_column
from pandas_df_commons.indexing.intersection import intersection_of_index
from pandas_df_commons.indexing.multiindex_utils import get_top_level_of_multi_index
from pandas_ml._abstract.transfromer import Transformer

# define default logger
_log = logging.getLogger(__name__)


# note top level rows and columns will be handled inside the function
def ml_features_labels(
        df: pd.DataFrame,
        feature_transformer: Transformer,
        label_transformer: Transformer,
        labels_shift=-1,
        replace_inf=np.nan
):
    if labels_shift > 0:
        _log.warning(f"Do you really want to shift positive? Like the past {labels_shift} values to the future ")

    features = ml_transform(df, feature_transformer, False)
    labels, label_inverter = ml_transform(df, label_transformer, True)

    features = features.replace([-np.inf, np.inf], replace_inf).dropna()
    labels = labels.shift(labels_shift).replace([-np.inf, np.inf], replace_inf).dropna()
    idx = intersection_of_index(features, labels)

    return features.loc[idx], labels.loc[idx], label_inverter


# note top level rows and columns will be handled inside the function
def ml_transform(df: pd.DataFrame, transformer: Transformer, return_inverter=False, parallel=False):
    """
    All transformers return a tuple of dataframe and an inverse function.
    Therefore, (in contrast to analytics functions) none of the transformers is decorated and each frame gets treated as
    a whole regardless if logically repeating data is present at the top level MultiIndex (like bar data of multiple
    stocks). In order to take such logically repeating datasets into account this function has to be used as the
    entrypoint - i.e.:

    transformed_df, inverter = df.ml.transform(
        SelectJoin(
            Select("Open", "High", "Low", "Close") >> GapUpperLowerBody() >> SelectJoin(
                Select("gap") >> LambertGaussianizer() >> Rescale(),
                Select("upper") >> LambertGaussianizer() >> Rescale(),
                Select("lower") >> LambertGaussianizer() >> Rescale(),
                Select("body") >> LambertGaussianizer() >> Rescale(),
            ),
            Select("Volume") >> PercentChange() >> LogNormalizer() >> Rescale()
        ),
        return_inverter=True
    )
    """

    def aggregator(axis):
        def agg(results, level=None):
            return (
                pd.concat([v[0] for v in results.values()], keys=results.keys(), axis=axis),
                pd.concat(
                    [v[1] if isinstance(v[1], pd.DataFrame) else pd.DataFrame({"T": [v[1]]}) for v in results.values()],
                    keys=results.keys(),
                    axis=axis
                )
            )
        return agg

    @foreach_top_level_row_and_column(parallel, row_aggregator=aggregator(0), column_aggregator=aggregator(1))
    def transform(_df):
        flows = deepcopy(transformer)
        return flows.transform(_df), flows

    # return transformed data frame with or without the fitted transformer
    df, fitted_transformers = transform(df)
    return (df, _Invertor(fitted_transformers)) if return_inverter else df


class Select(Transformer):

    def __init__(self, *columns, **kwargs) -> None:
        super().__init__(**kwargs)
        self.columns = list(columns)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = get_columns(df, self.columns)
        if df.ndim < 2: df = df.to_frame()
        return df

    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class SelectJoin(Transformer):

    def __init__(self, *selectors: Select, **kwargs) -> None:
        super().__init__(**kwargs)
        self.selectors = list(selectors)
        self.resulting_columns = None

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = [se(df) for se in self.selectors]
        self.resulting_columns = [f.columns.tolist() for f in dfs]

        return pd.concat(dfs, axis=1)

    def _inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = [se.inverse(df[self.resulting_columns[i]]) for i, se in enumerate(self.selectors)]
        return pd.concat(dfs, axis=1)


class _Invertor(object):

    def __init__(self, transformer: Transformer | pd.DataFrame):
        super().__init__()
        self.transformer = transformer

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        tl_rows, tl_cols = get_top_level_of_multi_index(df)

        def inverse_tl_column(trans_row, data_row):
            if tl_cols:
                data_cols = [trans_row[tc].iloc[0, 0].inverse(data_row[tc]) for tc in tl_cols]
            else:
                data_cols = [trans_row.iloc[0, 0].inverse(data_row)]

            return pd.concat(data_cols, keys=tl_cols, axis=1)

        if tl_rows:
            return pd.concat(
                [inverse_tl_column(self.transformer.loc[tlr], df.loc[tlr]) for tlr in tl_rows],
                keys=tl_rows,
                axis=0
            )
        elif tl_cols:
            return inverse_tl_column(self.transformer, df)
        else:
            return self.transformer.inverse(df)
