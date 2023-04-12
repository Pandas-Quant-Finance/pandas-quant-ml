from threading import Lock

from tensorflow import keras
import numpy as np
import pandas as pd

from pandas_df_commons.indexing._utils import get_top_level_rows
from .ml_traning_loop import training_loop


class KerasDfDataGenerator(keras.utils.Sequence):

    def __init__(
            self,
            df: pd.DataFrame,
            feature_columns,
            label_columns,
            batch_size=32,
            feature_look_back_window: int = None,
            label_look_back_window: int = None,
            shuffle=False,
            on_get_batch=None,
    ):
        super().__init__()
        self.df = df.dropna()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.feature_look_back_window = feature_look_back_window
        self.label_look_back_window = label_look_back_window
        self.on_get_batch = on_get_batch

        self.lock = Lock()
        self.length = self.__calc_length__()
        self.looper = None

        # initialize looper
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.length

    def __getitem__(self, index):
        # Generate data
        with self.lock:
            for i in range(2):
                # workaround for: https://github.com/keras-team/keras/issues/17946
                try:
                    X, y = next(self.looper)
                    break
                except StopIteration as e:
                    if i < 1:
                        self.__init_looper__()
                    else:
                        raise e

            if self.on_get_batch is not None:
                self.on_get_batch(index, X, y)

            return X, y

    def on_epoch_end(self):
        # print("on epoch end")
        with self.lock:
            self.__init_looper__()

    def __init_looper__(self):
        self.looper = training_loop(
            self.df,
            self.feature_columns,
            self.label_columns,
            self.batch_size,
            self.feature_look_back_window,
            self.label_look_back_window,
            shuffle=self.shuffle,
        )

    def __calc_length__(self):
        if self.label_look_back_window is not None:
            raise NotImplemented

        if self.feature_look_back_window is None:
            # ceil(Len(df)/batch size)
            length = len(self.df) / self.batch_size
        else:
            # ceil(for each TL row.index.apply(Len(i) - window size+1).sum / batch size )
            top_level_rows = get_top_level_rows(self.df)
            if top_level_rows:
                length = np.sum(
                    [self.df.loc[tlr].shape[0] - self.feature_look_back_window + 1 for tlr in top_level_rows]
                ) / self.batch_size
            else:
                length = (
                    self.df.shape[0] - self.feature_look_back_window + 1
                ) / self.batch_size

        return int(np.ceil(length))