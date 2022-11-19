"""Augment pandas DataFrame with methods for technical quant analysis"""
__version__ = '0.3.0'

from itertools import chain


class _ML(object):

    def __init__(self, df):
        from pandas_ml import analytics as anal
        from pandas_ml import transformers as trans
        from functools import partial, wraps

        self.df = df

        for name, func in chain(anal.__dict__.items(), trans.__dict__.items()):
            if name.startswith("ml_"):
                self.__dict__[name[3:]] = wraps(func)(partial(func, self.df))


def monkey_patch_dataframe(extender='ml'):
    from pandas.core.base import PandasObject

    existing = getattr(PandasObject, extender, None)
    if existing is not None:
        if not isinstance(existing.fget(None), _ML):
            raise ValueError(f"field already exists as {type(existing)}")

    setattr(PandasObject, extender, property(lambda self: _ML(self)))
