from abc import abstractmethod
from collections.abc import Callable
from typing import Iterable

import numpy as np

from pcmr.utils import ClassRegistry


class FeaturizerBase(Callable[[Iterable[str], np.ndarray]]):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        pass


FeaturizerRegistry = ClassRegistry()
