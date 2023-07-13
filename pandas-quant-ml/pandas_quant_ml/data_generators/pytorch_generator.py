from torch.utils.data import DataLoader

from pandas_quant_ml.utils.batch_cache import BatchCache


class TorchDataGenerator(DataLoader):

    def __init__(self, batch_cache: BatchCache):
        super().__init__(None)
        self.batch_cache = batch_cache

    def __iter__(self):
        pass

