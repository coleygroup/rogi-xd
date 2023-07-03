import logging
import warnings

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from rogi_xd.rogi import mask_inputs

from rogi_xd.utils.rogi import RogiKnnResult

logger = logging.getLogger(__name__)
    

def rogi_knn(
    X: ArrayLike,
    y: ArrayLike,
    normalize: bool = True,
    k: int = 5,
    **kwargs
) -> RogiKnnResult:
    X = np.array(X)
    y = np.array(y)

    if normalize:
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1))[:, 0]
    elif (y < 0).any() or (y > 1).any():
        warnings.warn("Input array 'y' has values outside [0, 1]. ROGI may be outside [0, 1]!")

    X, y_ = mask_inputs(X, y)
    if (n_invalid := len(y) - len(y_)) > 0:
        logger.info(f"Removed {n_invalid} input(s) with invalid features or scores")

    knn = KNeighborsRegressor(k).fit(X, y_)
    rmse = np.sqrt(np.mean(np.square(y_ - knn.predict(X))))

    return RogiKnnResult(rmse, None, len(y_))
