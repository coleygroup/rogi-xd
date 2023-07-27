from dataclasses import dataclass
from itertools import chain
import logging
from typing import Optional, Sequence, Union
import warnings

from fastcluster import complete as max_linkage
import numpy as np
from numpy.typing import ArrayLike
from rdkit.DataStructs import ExplicitBitVect, BulkTanimotoSimilarity, BulkDiceSimilarity
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import squareform, pdist
from rogi_xd.rogi import mask_inputs

from rogi_xd.utils import RogiResult
from rogi_xd.utils.rogi import Metric
from rogi_xd.utils.utils import flist

logger = logging.getLogger(__name__)


class DistanceMatrixCalculator:
    @classmethod
    def __call__(
        cls,
        X: Union[np.ndarray, Sequence[ExplicitBitVect]],
        metric: Union[str, Metric, None],
        max_dist: Optional[float] = None
    ):
        metric = None if metric is None else Metric.get(metric)

        if isinstance(X, np.ndarray):
            D = cls._calc_ndarray(X, metric, max_dist)
        elif isinstance(X[0], ExplicitBitVect):
            metric = Metric.get(metric) if metric is not None else Metric.TANIMOTO
            D = cls._calc_fps(X, metric)
        else:
            raise TypeError("arg 'X' must be of type `np.ndarray` | `Sequence[ExplicitBitVect]`")

        return D
    
    @classmethod
    def _calc_ndarray(
        cls, X: np.ndarray, metric: Optional[Metric], max_dist: Optional[float] = None
    ) -> np.ndarray:
        """Calculate the distance matrix of the input array X

        Raises
        ------
        ValueError
            if X has any invalid input values (e.g., 'inf' or 'nan')
        """
        metric = metric or Metric.EUCLIDEAN

        if not np.isfinite(X).all():
            raise ValueError("arg 'X' must have only finite vaues!")

        if metric == Metric.PRECOMPUTED:
            logging.info("Using precomputed distance matrix")
            if X.ndim == 1:
                D = X
            elif X.ndim == 2:
                D = squareform(X)
            else:
                raise ValueError(f"Precomputed distance matrix must have rank {{1, 2}}. got: {X.ndim}")

            max_dist_ = max_dist or 1.0
        else:
            D = pdist(X, metric.value)
            max_dist_ = max_dist or cls.estimate_max_dist(X, metric)

        # Scaling distances only normalizes the integration domain
        D = D / max_dist_

        if (D > 1.0).any():
            raise ValueError(
                "Pairwise distance matrix is not normalized! "
                f"Please ensure the provided 'd_max' is correct. got: {max_dist:0.3f}"
            )

        return D

    @classmethod
    def _calc_fps(
        cls, fps: Sequence[ExplicitBitVect], metric: Optional[Metric] = None, 
    ) -> np.ndarray:
        metric = metric or Metric.TANIMOTO

        logger.info("Computing distance matrix...")

        if metric == Metric.TANIMOTO:
            func = BulkTanimotoSimilarity
        elif metric == Metric.DICE:
            func = BulkDiceSimilarity
        else:
            raise ValueError(
                "Unsupported fingerprint distance metric!"
                f" got: {metric.value}. expected one of: (Metric.TANIMOTO, Metric.DICE)"
            )

        simss = (func(fps[i], fps[i + 1 :]) for i in range(len(fps)))
        sims = list(chain(*simss))
        S = np.array(sims)

        return 1 - S

    @classmethod
    def _estimate_max_dist(cls, X: np.ndarray, metric: Metric) -> float:
        EXPECTED_METRICS = {Metric.EUCLIDEAN, Metric.CITYBLOCK, Metric.COSINE, Metric.MAHALANOBIS}
        if metric not in EXPECTED_METRICS:
            raise ValueError(
                f"Can only estimate max distance for following metrics: {EXPECTED_METRICS}"
            )
        
        bounds = np.stack([X.min(0), X.max(0)])
        range_ = bounds[1] - bounds[0]

        if metric == Metric.EUCLIDEAN:
            max_dist = np.linalg.norm(range_, 2)
        elif metric == Metric.CITYBLOCK:
            max_dist = np.sum(range_)
        elif metric == Metric.COSINE:
            # scipy.pdist returns `1 - cosine` so that \in [0,2]
            max_dist = 2.0
        elif metric == Metric.MAHALANOBIS:
            # more general approach that would work for other metrics too excluding e.g. cosine
            CV = np.atleast_2d(np.cov(X.astype(np.double, copy=False).T))
            VI = np.linalg.inv(CV).T.copy()
            max_dist = pdist(bounds, metric="mahalanobis", VI=VI)[0]
        else:
            raise RuntimeError

        logger.debug(f"Estimated max dist: {max_dist:0.3f}")
        return max_dist


@dataclass
class ROGIcalculator:
    normalize: bool = True,
    min_dt: float = 0.01,
    nboots: int = 1,

    @classmethod
    def nmoment(
        cls, x: ArrayLike, n: int = 2, c: Optional[float] = None, w: Optional[ArrayLike] = None
    ) -> float:
        """calculate the (weighted) n-th moment

        Parameters
        ----------
        x : array_like
            samples
        n : int, default=2
            the moment to calculate
        c : Optional[float], default=None
            central location. If None, the average of `x` is used
        w : Optional[ArrayLike], default=None
            sample weights, if weighted moment is to be computed.

        Returns
        -------
        float
            the n-th moment
        """
        x = np.array(x)
        c = c if c is not None else np.average(x, weights=w)

        return np.average((x - c) ** n, weights=w)

    @classmethod
    def coarsened_sd(cls, y: np.ndarray, Z: np.ndarray, t: float) -> tuple[float, int]:
        """the coarsened standard deviation of the samples `y`

        The coarsened moment is calculated via clustering the input samples `y` according to the input linkage matrix `Z` and distance threhsold `t` and calculating the mean value of each cluster. The 2nd weighted moment (variance) of these cluster means is calculated with weights equal to the size of the respective cluster.

        NOTE: The samples are assumed to lie in the range [0, 1], so the coarsened standard deviation is multiplied by 2 to normalize it to the range [0, 1].

        Parameters
        ----------
        y : np.ndarray
            the original samples
        Z : np.ndarray
            the linkage matrix from hierarchical cluster. See :func:`scipy.cluster.hierarchy.linkage`
            for more details
        t : float
            the distance threshold to apply when forming clusters

        Returns
        -------
        float
            the coarsened standard deviation
        int
            the number of clusters at the coarse graining step
        """
        if (y < 0).any() or (y > 1).any():
            warnings.warn("Input array 'y' has values outside of [0, 1]")

        cluster_ids = fcluster(Z, t, "distance")
        clusters = set(cluster_ids)

        means = []
        weights = []
        for i in clusters:
            mask = cluster_ids == i
            means.append(y[mask].mean())
            weights.append(len(y[mask]))

        # max std dev is 0.5 --> multiply by 2 so that results is in [0,1]
        var = cls.nmoment(means, 2, weights=weights)
        sd_normalized = 2 * np.sqrt(var)

        return sd_normalized, len(clusters)


    def coarse_grain(
        self, D: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.debug("Clustering...")
        Z = max_linkage(D)
        all_distance_thresholds = Z[:, 2]

        logger.debug(f"Subsampling with minimum step size of {self.min_dt:0.3f}")
        thresholds = []
        t_prev = -1
        for t in all_distance_thresholds:
            if t < t_prev + self.min_dt:
                continue

            thresholds.append(t)
            t_prev = t

        logger.debug(f"Coarsening with thresholds {flist(thresholds):0.3f}")
        cg_sds, n_clusters = zip(*[self.coarsened_sd(y, Z, t) for t in thresholds])

        # when num_clusters == num_data ==> stddev/skewness of dataset
        thresholds = np.array([0.0, *thresholds, 1.0])
        cg_sds = np.array([2 * y.std(), *cg_sds, 0])
        n_clusters = np.array([len(y), *n_clusters, 1])

        return thresholds, cg_sds, n_clusters