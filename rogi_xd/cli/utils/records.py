from dataclasses import InitVar, dataclass, field
from typing import NamedTuple

import numpy as np

from rogi_xd.utils.rogi import RogiResult


@dataclass(frozen=True)
class RogiResultMixin:
    rogi_result: InitVar[RogiResult]

    rogi: float = field(init=False)
    n_valid: int = field(init=False)
    thresholds: np.ndarray = field(init=False)
    cg_sds: np.ndarray = field(init=False)
    n_clusters: np.ndarray = field(init=False)

    def __post_init__(self, rogi_result: RogiResult):
        for k, v in rogi_result._asdict().items():
            if k == "uncertainty":
                continue
            object.__setattr__(self, k, v)


@dataclass(frozen=True)
class RogiRecordBase:
    representation: str
    dataset_and_task: str


@dataclass(frozen=True)
class RogiRecord(RogiResultMixin, RogiRecordBase):
    pass


@dataclass(frozen=True)
class RogiKnnRecordBase(RogiRecordBase):
    k: int

@dataclass(frozen=True)
class RogiKnnRecord(RogiResultMixin, RogiKnnRecordBase):
    pass

class CrossValdiationResult(NamedTuple):
    model: str
    r2: float
    rmse: float
    mae: float


@dataclass(frozen=True)
class RogiAndCrossValRecord(RogiRecord):
    cv_result: InitVar[CrossValdiationResult]

    model: str = field(init=False)
    r2: float = field(init=False)
    rmse: float = field(init=False)
    mae: float = field(init=False)

    def __post_init__(self, rogi_result: RogiResult, cv_result: CrossValdiationResult):
        super().__post_init__(rogi_result)
        for k, v in cv_result._asdict().items():
            object.__setattr__(self, k, v)
