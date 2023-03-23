from dataclasses import InitVar, dataclass, field
from typing import NamedTuple

import numpy as np

from pcmr.utils.rogi import RogiResult


@dataclass(frozen=True)
class RogiRecord:
    representation: str
    dataset_and_task: str
    rr: InitVar[RogiResult]

    rogi: float = field(init=False)
    n_valid: int = field(init=False)
    thresholds: np.ndarray = field(init=False)
    cg_sds: np.ndarray = field(init=False)
    n_clusters: np.ndarray = field(init=False)

    def __post_init__(self, rr: RogiResult):
        for k, v in rr._asdict().items():
            if k == "uncertainty":
                continue
            object.__setattr__(self, k, v)


class CrossValdiationResult(NamedTuple):
    model: str
    r2: float
    rmse: float
    mae: float


@dataclass(frozen=True)
class RogiAndCrossValRecord(RogiRecord):
    cvr: InitVar[CrossValdiationResult]

    model: str = field(init=False)
    r2: float = field(init=False)
    rmse: float = field(init=False)
    mae: float = field(init=False)

    def __post_init__(self, rr: RogiResult, cvr: CrossValdiationResult):
        super().__post_init__(rr)
        for k, v in cvr._asdict().items():
            object.__setattr__(self, k, v)
