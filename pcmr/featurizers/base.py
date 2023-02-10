from abc import abstractmethod
from collections.abc import Callable
from typing import Iterable
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike

from pcmr.utils import ClassRegistry

FeaturizerRegistry = ClassRegistry()


class FeaturizerBase(Callable[[Iterable[str], np.ndarray]]):
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
    
    @abstractmethod
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        pass

    @abstractmethod
    def finetune(self, smis: Iterable[str], targets: ArrayLike) -> Self:
        """fine tune this featurizer on the input data"""