from enum import auto
from typing import NamedTuple, Optional

import numpy as np

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


class IntegrationDomain(AutoName):
    THRESHOLD = auto()
    CLUSTER_RATIO = auto()
    LOG_CLUSTER_RATIO = auto()


class FingerprintConfig(NamedTuple):
    fp: Fingerprint = Fingerprint.MORGAN
    radius: int = 2
    length: int = 2048


class RogiResult(NamedTuple):
    rogi: float
    uncertainty: Optional[float]
    n_valid: int
    thresholds: np.ndarray
    cg_sds: np.ndarray
    n_clusters: np.ndarray
