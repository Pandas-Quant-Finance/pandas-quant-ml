import inspect
from typing import Iterable, Tuple


def make_iterable(ele) -> Iterable[Tuple]:
    if inspect.isgenerator(ele):
        return ele
    elif inspect.isgeneratorfunction(ele):
        return ele()
    if isinstance(ele, dict):
        return ele.items()
    elif not isinstance(ele, Iterable):
        return [(None, ele)]
    else:
        # raise ValueError("Should never get here")
        return ele



