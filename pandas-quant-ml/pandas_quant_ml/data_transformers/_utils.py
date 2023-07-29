from typing import Iterable, Callable


def renaming_columns(names: str | Iterable[str] | Callable[[str, int], str] = None):
    def rename(x, i):
        if isinstance(names, Iterable) and not isinstance(names, str):
            return f"{x}_{names[i]}"
        else:
            return names

    return rename
