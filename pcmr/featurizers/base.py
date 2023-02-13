from abc import ABC, abstractmethod
from typing import Iterable
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike

from pcmr.utils import ClassRegistry

FeaturizerRegistry = ClassRegistry()


class FeaturizerBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        """featurize the input SMILES strings into vectors"""

    @abstractmethod
    def finetune(self, smis: Iterable[str], targets: ArrayLike) -> Self:
        """fine tune this featurizer on the input data"""
