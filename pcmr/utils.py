from __future__ import annotations

from enum import Enum, auto
from typing import Union


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


class Dataset(AutoName):
    BACE = auto()
    CLEARANCE = auto()
    CLINTOX = auto()
    DELANEY = auto()
    FREESOLV = auto()
    LIPO = auto()
    PDBBIND = auto()
    