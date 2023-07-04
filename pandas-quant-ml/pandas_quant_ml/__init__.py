"""Augment pandas DataFrame with methods for technical quant analysis"""
__version__ = '0.3.0'

from functools import partial as _partial

from pandas_df_commons._utils.patching import _monkey_patch_dataframe, _add_functions

_ML = _add_functions(
    'pandas_quant_ml.analytics', # TODO later 'pandas_quant_ml.transformers',
    filter=lambda _, x: x[3:] if x.startswith("ml_") else None
)

monkey_patch_dataframe = _partial(_monkey_patch_dataframe, extension_default_value='ml', extension_class=_ML)
