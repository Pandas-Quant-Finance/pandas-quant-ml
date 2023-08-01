from string import Template
from typing import Iterable, Callable


def renaming_columns(names: str | Iterable[str] | Callable[[str, int], str] | Template = None):
    def rename(x, i, p):
        if isinstance(names, Iterable) and not isinstance(names, str):
            return f"{x}_{names[i]}"
        elif isinstance(names, Template):
            return names.substitute(x=x, i=i, p=p)
        else:
            return names

    return rename
