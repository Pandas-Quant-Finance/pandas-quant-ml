from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Type, Any, Tuple, Dict, List, Generator, Callable

import numpy as np
import pandas as pd

from pandas_df_commons._utils.streaming import frames_at_common_index
from pandas_df_commons.indexing._utils import get_top_level_rows
from pandas_df_commons.indexing.multiindex_utils import make_top_level_row_iterator, loc_at_level, last_index_value, \
    nth, unique_level_values, index_shape
from pandas_quant_ml.data_transformers.data_transformer import DataTransformer
from pandas_quant_ml.data_transformers.generic.constant import DataConstant
from pandas_quant_ml.utils.batch_cache import BatchCache, MemCache
from pandas_quant_ml.utils.iter_utils import make_iterable
from pandas_quant_ml.utils.split_frame import split_frames, get_split_index


_LOG = logging.getLogger(__name__)


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
        
        Pandas dtype	Python type	NumPy type	Usage
        object	        str or mixed	string_, unicode_, mixed types	Text or mixed numeric and non-numeric values
        int64	        int	int_, int8, int16, int32, int64, uint8, uint16, uint32, uint64	Integer numbers
        float64	        float	float_, float16, float32, float64	Floating point numbers
        bool	        bool	bool_	True/False values
        datetime64	    NA	datetime64[ns]	Date and time values
        timedelta[ns]	NA	NA	Differences between two datetimes
        category	    NA	NA	Finite list of text values
        
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
            sample_weights: DataTransformer = None,
            train_test_split_ratio: float | Tuple[float, float] = 0.75,
            batch_size: int = None,
            include_frame_name_category: bool | int = False,
            feature_shape: Tuple[int, ...] = None,
            label_shape: Tuple[int, ...] = None,
            batch_cache: Type[BatchCache] = MemCache,
    ):
        self.feature_pipeline = feature_pipeline
        self.label_pipeline = label_pipeline
        self.sample_weights = sample_weights
        self.train_test_split_ratio = (train_test_split_ratio, 0) if isinstance(train_test_split_ratio, float) else train_test_split_ratio
        self.batch_size = batch_size
        self.include_frame_name_category = include_frame_name_category
        self.feature_shape = feature_shape
        self.label_shape = label_shape
        self.batch_cache = batch_cache

        self._feature_pipelines = defaultdict(lambda: deepcopy(self.feature_pipeline))
        self._label_pipelines = defaultdict(lambda: deepcopy(self.label_pipeline))
        self._sample_weights_pipelines = defaultdict(lambda: deepcopy(self.sample_weights) if self.sample_weights is not None else None)
        self._split_indices = {}
        self._meta_data: MetaData = None

    @property
    def meta_data(self):
        assert self._meta_data is not None, "train test iterator need to be constructed first"
        return self._meta_data

    def train_test_iterator(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            nth_row_only: int = None,
    ) -> Tuple[BatchCache, BatchCache] | Tuple[BatchCache, BatchCache, BatchCache]:
        train_cache, val_cache, test_cache = self.batch_cache(), self.batch_cache(), self.batch_cache()
        categorical, real = None, None
        categories = defaultdict(set)
        target = -1

        for train_val_test in self._train_test_batches(frames, nth_row_only):
            train_val_test = train_val_test if self.train_test_split_ratio[1] > 0 else train_val_test[:-1]
            caches = [train_cache, val_cache, test_cache] if self.train_test_split_ratio[1] > 0 else [train_cache, test_cache]
            for cache, (features, labels, weights) in zip(caches, train_val_test):
                for b in zip(features, labels, weights):
                    # collect some meta data
                    if target < 0: target = b[1].shape[1]
                    if real is None: real = [i for i, dt in enumerate(b[0].dtypes) if dt == 'float']
                    if categorical is None: categorical = [i for i, dt in enumerate(b[0].dtypes) if dt != 'float']
                    for c in categorical: categories[c].update(pd.unique(b[0].iloc[:, c]))

                    features_in_shape = self.get_features_in_shape(b[0])
                    labels_in_shape = self.get_labels_in_shape(b[1])
                    index_in_shape = b[0].index.values.reshape(features_in_shape[0].shape[:-1])
                    cache.add_batch(
                        index_in_shape,
                        *features_in_shape, # TODO Later we want to allow tuple numpy arrays for different fetatures/labels like int, float
                        *labels_in_shape,
                        b[2].values.reshape(labels_in_shape[0].shape[:-1]),
                    )

        self._meta_data = MetaData(
            len(categorical) + len(real),
            target,
            real,
            categorical,
            [len(c) for c in categories.values()],
        )

        # return iterable AND sub_scriptable object batch_cache[123]
        return (train_cache, val_cache, test_cache) if self.train_test_split_ratio[1] > 0 else (train_cache, test_cache)

    def get_features_in_shape(self, features: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        shape = (index_shape(features) + index_shape(features, axis=1)) if self.feature_shape is None else self.feature_shape
        shape = tuple(features.shape[1] if s == -2 else s for s in shape)
        return features.values.reshape(shape),

    def get_labels_in_shape(self, labels: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        shape = (index_shape(labels) + index_shape(labels, axis=1)) if self.label_shape is None else self.label_shape
        shape = tuple(labels.shape[1] if s == -2 else s for s in shape)
        return labels.values.reshape(shape),

    def _train_test_batches(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            nth_row_only: int = None,
    ) -> Generator[Tuple[Tuple[Batch, Batch, Batch], ...], None, None]:
        for name, df in make_top_level_row_iterator(make_iterable(frames)):
            data_length = len(unique_level_values(df))
            test_length = int(data_length - data_length * self.train_test_split_ratio[0])
            feature_df, _ = self._feature_pipelines[name].fit_transform(df, test_length)
            label_df, _ = self._label_pipelines[name].fit_transform(df, test_length)
            if self._sample_weights_pipelines[name] is not None:
                weight_df, _ = self._sample_weights_pipelines[name].fit_transform(df, 0)
            else:
                weight_df = label_df[[]].assign(w=1.0)

            # add categorical variable for frame name if requested
            feature_df = self._add_frame_name_category(name, feature_df)

            # fix leaking test data into training data by fixing the test data length
            level = 0  #if feature_df.index.nlevels != label_df.index.nlevels else None
            predicted_periods = len(pd.unique(loc_at_level(feature_df, slice(last_index_value(label_df, 0), None, None), level).index.get_level_values(0))) - 1
            test_length -= predicted_periods

            # align frames at common index because of different length after data pipelines
            feature_df, label_df = frames_at_common_index(feature_df, label_df, level=level)
            weight_df = weight_df.loc[label_df.index if weight_df.index.nlevels == label_df.index.nlevels else get_top_level_rows(label_df)]

            # find index to split data into train and test set
            split_idx = get_split_index(feature_df, test_length)
            self._split_indices[name] = (split_idx, split_idx)

            # if skip data, i.e. if no overlapping data should be used for training from moving windows
            if nth_row_only is not None:
                feature_df = nth(feature_df, nth_row_only, level=0)
                label_df = nth(label_df, nth_row_only, level=0)
                weight_df = nth(weight_df, nth_row_only, level=0)

            # split training and test/validation data
            (feature_train_df, label_train_df, weight_train_df), (feature_test_df, label_test_df, weight_test_df) =\
                split_frames(feature_df, label_df, weight_df, split_index=split_idx)
            _LOG.info(f"Train test split for {name}: {len(feature_train_df)} / {len(feature_test_df)}")

            if self.train_test_split_ratio[1] > 0:
                data_length = len(unique_level_values(label_test_df))
                test_length = int(data_length - data_length * self.train_test_split_ratio[1])
                self._split_indices[name] = (self._split_indices[name][0], label_test_df.index[-test_length])
                (feature_val_df, label_val_df, weight_val_df), (feature_test_df, label_test_df, weight_test_df) =\
                    split_frames(feature_test_df, label_test_df, weight_test_df, test_length=test_length)
            else:
                feature_val_df, label_val_df, weight_val_df = feature_test_df, label_test_df, weight_test_df

            # make batch generators and yield them
            yield (
                tuple(Batch(f, self.batch_size) for f in [feature_train_df, label_train_df, weight_train_df]),
                tuple(Batch(f, self.batch_size) for f in[feature_val_df, label_val_df, weight_val_df]),
                tuple(Batch(f, self.batch_size) for f in[feature_test_df, label_test_df, weight_test_df]),
            )

    def inference_generator(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            predictor: Callable[[np.ndarray], np.ndarray],
            include_labels: bool = False,
    ):
        def wrapped_predictor(batches) -> pd.DataFrame:
            predicted_dfs = []
            for b in batches:
                values = self.get_features_in_shape(b)
                predicted = predictor(*values)
                predicted_dfs.append(pd.DataFrame(predicted.reshape(len(b.index), -1), index=b.index))

            return pd.concat(predicted_dfs, axis=0)

        for name, df in self._inference_generator(frames, wrapped_predictor, include_labels):
            yield name, df

    def _inference_generator(
            self,
            frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]] | Dict[Any, pd.DataFrame],
            predictor: Callable[[Batch], pd.DataFrame],
            include_labels: bool = False,
    ):
        for name, df in make_top_level_row_iterator(make_iterable(frames)):
            feature_df, _ = self._feature_pipelines[name].transform(df, False)
            labels_df, queue = self._label_pipelines[name].transform(df)

            # add categorical variable for frame name if requested
            feature_df = self._add_frame_name_category(name, feature_df)

            batcher = Batch(feature_df, self.batch_size or len(feature_df))
            predicted_df = predictor(batcher)

            # fix prediction column names. Note that it can be that the coulmn count is different
            # i.e. when using a float as label but the model predicts a pobability distribution
            if len(predicted_df.columns) == len(labels_df.columns):
                predicted_df.columns = labels_df.columns

            try:
                predicted_df = self._label_pipelines[name].inverse(predicted_df, queue)
            except NotImplementedError as e:
                _LOG.warning(e)

            if include_labels:
                predicted_df = predicted_df.join(labels_df, how='outer', rsuffix='_TRUE')
                predicted_df["_SAMPLE_"] = 'TRAIN'
                if self._split_indices[name][0] is not None:
                    predicted_df.loc[self._split_indices[name][0]:, "_SAMPLE_"] = 'VAL'
                    predicted_df.loc[self._split_indices[name][1]:, "_SAMPLE_"] = 'TEST'

            yield name, predicted_df

    def _add_frame_name_category(self, name, df):
        if self.include_frame_name_category:
            category = next(i for i, k in enumerate(self._feature_pipelines.keys()) if k == name)
            pos = self.include_frame_name_category if isinstance(self.include_frame_name_category, int) else 0
            df.insert(df.shape[1] + pos + 1 if pos < 0 else pos, "frame_name_category", category)

        return df


@dataclass
class MetaData(object):
    input_size: int
    output_size: int
    known_regular_inputs: List[int]      # locations of float data features and constants
    known_categorical_inputs: List[int]  # locations of categorical data features can constants
    category_counts: List[int]  # list of counts of categories of each categorical variable


class Batch(object):

    def __init__(self, df: pd.DataFrame, batch_size: int):
        self.df = df
        self.batch_size = len(df) if batch_size is None else batch_size

        self._current_index = -1
        self._index = get_top_level_rows(df) if isinstance(df.index, pd.MultiIndex) else df.index

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index >= len(self._index): raise StopIteration()

        try:
            return self.df.loc[self._index[self._current_index:self._current_index + self.batch_size]]
        finally:
            self._current_index += self.batch_size
