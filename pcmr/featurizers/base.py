from abc import ABC, abstractmethod
from typing import Iterable
from typing_extensions import Self

import numpy as np
from numpy.typing import ArrayLike

from pcmr.utils import ClassRegistry
from pcmr.utils.rogi import Metric

FeaturizerRegistry = ClassRegistry()


class FeaturizerBase(ABC):
    metric = Metric.EUCLIDEAN
    """the distance metric to use with this featurizer"""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        """featurize the input SMILES strings into vectors"""

    @abstractmethod
    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        """finetune this featurizer on the input data

        Parameters
        ----------
        *splits : Iterable[tuple[Iterable[str], ArrayLike]]
            an iterable of splits, where a "split" is a tuple of SMILES strings and associated
            property values. Depending on the number of splits provided, the input will be
            interepreted differently

        1. a single split that will be further split 90/10 into train/val data.
        2. train and val splits.
        3. train, val, and test splits. NOTE: the test split will be ignored.
        """

    def __str__(self) -> str:
        return self.alias