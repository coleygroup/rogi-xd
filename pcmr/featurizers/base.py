from collections.abc import Callable
from typing import Iterable

import numpy as np

class Featurizer(Callable[[Iterable[str], np.ndarray]]):
    pass