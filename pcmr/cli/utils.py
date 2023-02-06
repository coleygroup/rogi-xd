from enum import auto
import functools
from typing import Any, Callable, Iterable, NamedTuple, Optional

from tdc.utils import fuzzy_search

from pcmr.utils import AutoName


class RogiCalculationResult(NamedTuple):
    featurizer: str
    dataset_and_task: str
    n_valid: int
    rogi: float


class ModelType(AutoName):
    GIN = auto()
    VAE = auto()


def bounded(lo=None, hi=None):
    def decorator(f):
        if lo is None and hi is None:
            raise ValueError("No bounds provided!")

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            x = f(*args, **kwargs)

            if (lo is not None and hi is not None) and not lo <= x <= hi:
                raise ValueError(f"Parsed value outside of range [{lo}, {hi}]! got: {x}")
            if hi is not None and x > hi:
                raise ValueError(f"Parsed value below {hi}! got: {x}")
            if lo is not None and x < lo:
                raise ValueError(f"Parsed value above {lo}]! got: {x}")
            return x

        return wrapper

    return decorator


def dataset_and_task(value) -> tuple[str, Optional[str]]:
    tokens = value.split("/")

    if len(tokens) == 1:
        dataset, task = tokens[0], None
    elif len(tokens) == 2:
        dataset, task = tokens
    else:
        raise ValueError("value must be of form: ('A', 'A/B')")

    dataset = dataset.upper()
    task = task or None

    return dataset, task


def fuzzy_lookup(choices: list[str]) -> Callable[[str], str]:
    def fun(choice: str):
        try:
            return fuzzy_search(choice, choices)
        except ValueError:
            return ValueError(f"Invalid choice! '{choice}' is not in possible choices: {choices}")

    return fun


def pop_attr(o: object, attr: str, *args) -> Optional[Any]:
    """like `pop()` but for attribute maps"""
    if len(args) == 0:
        return _pop_attr(o, attr)
    if len(args) == 1:
        default = args[0]
        return _pop_attr_d(o, attr, default)

    raise TypeError(f"Expected at most 3 arguments! got: {len(args)}")


def _pop_attr(o: object, attr: str) -> Any:
    val = getattr(o, attr)
    delattr(o, attr)

    return val


def _pop_attr_d(o: object, attr: str, default: Optional[Any] = None) -> Optional[Any]:
    try:
        val = getattr(o, attr)
        delattr(o, attr)
    except AttributeError:
        val = default

    return val
