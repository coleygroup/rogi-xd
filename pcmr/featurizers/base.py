from collections.abc import Callable
from typing import Iterable

import numpy as np

from pcmr.utils import ClassRegistry


class Featurizer(Callable[[Iterable[str], np.ndarray]]):
    pass


FeaturizerRegistry = ClassRegistry()