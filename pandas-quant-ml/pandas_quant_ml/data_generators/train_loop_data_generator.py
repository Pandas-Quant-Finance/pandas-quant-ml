from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Iterable, Type, Any, Tuple, Dict

import pandas as pd

from pandas_df_commons._utils.streaming import frames_at_common_index
from pandas_df_commons.indexing._utils import get_top_level_rows
from pandas_df_commons.indexing.multiindex_utils import make_top_level_row_iterator
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer
from pandas_quant_ml.utils.batch_cache import BatchCache, MemCache
from pandas_quant_ml.utils.iter_utils import make_iterable

"""
   TODO  
        -> add  sklearn.preprocessing.LabelEncoder() to transfomer pipeline
        
        -> add meta information like: (this would have to be done in the Transformer class) 
            "input_size": input_size,
            "output_size": len(_get_locations({InputTypes.TARGET}, self._column_definition)),
            "category_counts": self._num_classes_per_cat_input,
            "static_input_loc": _get_locations({InputTypes.STATIC_INPUT}, column_definition),
            "known_regular_inputs": _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, real_inputs),
            "known_categorical_inputs": _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, categorical_inputs),
        
    Emulate the transformer output:
        col_mappings = {
            "identifier": [id_col],
            "date": [time_col],
            "inputs": input_cols,
            "outputs": [target_col],
        }
        => this should be obsolete as we want to use labels_pipeline -> inference -> inverse of labels_pipeline
         
"""

class TrainTestLoop(object):

    def __init__(
            self,
            feature_pipeline: DataTransformer,
            label_pipeline: DataTransformer,
            train_test_split_ratio: float = 0.75,
            batch_size: int = None,
            look_back_window: int = None,
            batch_cache: Type[BatchCache] = MemCache,
    ):
        self.feature_pipeline = feature_pipeline
        self.label_pipeline = label_pipeline
        self.train_test_split_ratio = train_test_split_ratio
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.batch_cache = batch_cache

        self._current_epoch = -1
        self._feature_pipelines = defaultdict(lambda: deepcopy(self.feature_pipeline))
        self._label_pipelines = defaultdict(lambda: deepcopy(self.label_pipeline))

    def train_test_iterator(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
    ) -> Tuple[BatchCache, BatchCache]:
        train_cache, test_cache = self.batch_cache(), self.batch_cache()

        for name, df in make_top_level_row_iterator(make_iterable(frames)):
            test_length = int(len(df) - len(df) * self.train_test_split_ratio)
            feature_df = self._feature_pipelines[name].fit_transform(df, test_length)
            label_df = self._label_pipelines[name].fit_transform(df, test_length)
            predicted_periods = len(feature_df.loc[label_df.index[-1]: ])

            # TODO store meta information like input size, output size, etc.

            # split training and test/validation data
            feature_df, label_df = frames_at_common_index(feature_df, label_df)

            # fix leaking test data into training data
            test_length -= predicted_periods
            feature_train_df, label_train_df = feature_df.iloc[:len(df)-test_length], label_df.iloc[:len(df)-test_length]
            if test_length <= 0:
                feature_test_df, label_test_df = None, None
            else:
                feature_test_df, label_test_df = feature_df.iloc[-test_length:], label_df.iloc[-test_length:]

            # make batches
            bs = len(df) if self.batch_size is None else self.batch_size
            for cache, features, labels in [
                (train_cache, feature_train_df, label_train_df,),
                (test_cache, feature_test_df, label_test_df, )
            ]:
                for b in zip(Batch(features, bs), Batch(labels, bs)):
                    cache.add_batch(b[0].index.values, b[0].values, b[1].values, b[2].values)

        # TODO fit sklearn.preprocessing.LabelEncoder()
        #   for self._feature_pipelines.keys()

        # return iterable AND sub_scriptable object batch_cache[123]
        return train_cache, test_cache

    def as_inference_generator(self):
        # TODO we need to return an object that we can pickle and re-use with new data frames
        pass


class Batch(object):

    def __init__(self, df: pd.DataFrame, batch_size: int):
        self.df = df
        self.batch_size = batch_size

        self._current_index = -1
        self._index = get_top_level_rows(df) if isinstance(df.index, pd.MultiIndex) else df.index

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        try:
            return self.df.loc[self._index[self._current_index:self._current_index + self.batch_size]]
        finally:
            self._current_index += self.batch_size
