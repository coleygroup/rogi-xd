from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike

from rogi_xd.featurizers.base import FeaturizerBase, FeaturizerRegistry
from rogi_xd.utils.rogi import Metric


@FeaturizerRegistry.register(alias="random")
class RandomFeauturizer(FeaturizerBase):
    metric = Metric.EUCLIDEAN

    def __init__(self, length: Optional[int] = 128, **kwargs):
        self.length = length or 128

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        n = sum(1 for _ in smis)

        return np.random.rand(n, self.length)

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        return self

    def __str__(self) -> str:
        return f"{self.alias}{self.length}"
