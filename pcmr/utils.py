from __future__ import annotations

from enum import Enum, auto

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    @classmethod
    def get(cls, alias: str) -> AutoName:
        try:
            return cls[alias.upper()]
        except KeyError:
            raise ValueError(f"Unsupported {cls.__name__} alias! got: {alias}")


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
