from typing import Iterable, Callable, Tuple

import pandas as pd


def make_iterable(ele) -> Iterable[Tuple]:
    if isinstance(ele, dict):
        return ele.items()
    elif not isinstance(ele, Iterable):
        return [(None, ele)]
    else:
        return ele



