from typing import NamedTuple, Optional


class RogiCalculationResult(NamedTuple):
    featurizer: str
    dataset: str
    task: Optional[str]
    n_valid: int
    rogi: float
