from enum import auto
from typing import NamedTuple
from pcmr.utils.utils import AutoName


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