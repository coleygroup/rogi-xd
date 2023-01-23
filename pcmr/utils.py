from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto
import functools
from typing import Iterator, NamedTuple, Optional, Type, Union


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self) -> str:
        return self.value
        
    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unsupported {cls.__name__} alias! got: {name}")

    @classmethod
    def keys(cls) -> list[str]:
        return [e.value for e in cls]


class Fingerprint(AutoName):
    MORGAN = auto()
    TOPOLOGICAL = auto()


class Metric(AutoName):
    DICE = auto()
    TANIMOTO = auto()
    EUCLIDEAN = auto()
    COSINE = auto()
    CITYBLOCK = auto()
    MAHALANOBIS = auto()
    PRECOMPUTED = auto()


class FingerprintConfig(NamedTuple):
    fp: Fingerprint = Fingerprint.MORGAN
    radius: int = 2
    length: int = 2048


class ClassRegistry(Mapping[str, Type]):
    __registry = {}

    def register(self, cls=None, *, alias: Optional[str] = None):
        def actual_decorator(cls):
            key = alias or cls.__name__.lower()
            self.__registry[key] = cls

            @functools.wraps(cls)
            def cls_wrapper(*args, **kwargs):
                return cls(*args, **kwargs)

            return cls_wrapper

        if cls:
            return actual_decorator(cls)

        return actual_decorator

    __call__ = register
    def __getitem__(self, key: str) -> Type: self.__registry[key.lower()]
    def __iter__(self) -> Iterator[str]: return iter(self.__registry)
    def __len__(self) -> int: return len(self.__registry)