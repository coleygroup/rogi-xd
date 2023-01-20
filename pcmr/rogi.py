from itertools import chain
import logging
from typing import Iterable, Optional, Union
import warnings

from fastcluster import complete as max_linkage
import numpy as np
from numpy.typing import ArrayLike
from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs import ExplicitBitVect, BulkTanimotoSimilarity, BulkDiceSimilarity
from scipy.cluster.hierarchy import fcluster
from scipy.integrate import trapezoid
from scipy.spatial.distance import squareform, pdist

from pcmr.utils import Fingerprint, Metric

logger = logging.getLogger(__name__)


def safe_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if x.max() - x.min() > eps:
        return (x - x.min()) / (x.max() - x.min())

    return x - x.min()


def unsquareform(A: np.ndarray) -> np.ndarray:
    """Return the condensed distance matrix (upper triangular) of the square distance matrix `A`"""
    return A[np.triu_indices(A.shape[0], k=1)]


def calc_max_dist(X: np.ndarray, metric: Metric) -> float:
    bounds = np.stack([X.min(0), X.max(0)])
    ranges = np.ptp(bounds, 0)

    if metric == Metric.EUCLIDEAN:
        max_dist = np.sqrt(np.sum(ranges**2))
    elif metric == Metric.CITYBLOCK:
        max_dist = np.sum(ranges)
    elif metric == Metric.COSINE:
        # in scipy pdist, 1-cosine is returned so that \in [0,2]
        max_dist = 2.0
    elif metric == Metric.MAHALANOBIS:
        # more general approach that would work for other metrics too excluding e.g. cosine
        CV = np.atleast_2d(np.cov(X.astype(np.double, copy=False).T))
        VI = np.linalg.inv(CV).T.copy()
        max_dist = pdist(bounds, metric="mahalanobis", VI=VI)[0]
    else:
        raise ValueError(f"Invalid metric! got: {metric.value}")

    return max_dist


def calc_distance_matrix_X(
    X: np.ndarray, metric: Metric, max_dist: Optional[float] = None
) -> np.ndarray:
    if metric == Metric.PRECOMPUTED:
        if X.ndim == 1:
            D = X
        elif X.ndim == 2:
            D = unsquareform(X)
        else:
            raise ValueError(f"Precomputed distance matrix have have rank {{1, 2}}. got: {X.ndim}")

        max_dist = max_dist or 1.0
    else:
        D: np.ndarray = pdist(X, metric.value)
        max_dist = max_dist or calc_max_dist(X, metric)

    # Scaling distances only normalizes the integration domain 
    D = D / max_dist

    if (D > 1.0).any():
        raise ValueError(
            "Pairwise distance matrix is not normalized! "
            f"Please ensure the provided 'max_dist' is correct. got: {max_dist:0.3f}"
        )

    return D


def calc_fps(
    mols: Iterable[Chem.Mol],
    fp: Fingerprint = Fingerprint.MORGAN,
    radius: int = 2,
    length: int = 2048,
) -> list[ExplicitBitVect]:
    if fp == Fingerprint.MORGAN:
        fps = [Chem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=length) for m in mols]
    elif fp == Fingerprint.TOPOLOGICAL:
        fps = [Chem.RDKFingerprint(m) for m in mols]
    else:
        raise ValueError(f"Invalid fingerprint! got: {fp.value}")

    return fps


def calc_distance_matrix_fps(fps, metric: Metric = Metric.TANIMOTO) -> np.ndarray:
    logger.info("Computing distance matrix...")

    if metric == Metric.TANIMOTO:
        sim_func = BulkTanimotoSimilarity
    elif metric == Metric.DICE:
        sim_func = BulkDiceSimilarity
    else:
        raise ValueError(f"Unsupported metric! got: {metric.value}")

    simss = (sim_func(fps[i], fps[i + 1 :]) for i in range(len(fps)))
    sims = list(chain(*simss))
    S = np.array(sims)

    return 1 - S


def validate_smis(smis: Iterable[str]) -> list[str]:
    canon_smis = []
    for smi in smis:
        try:
            c_smi = Chem.CanonSmiles(smi)
            canon_smis.append(c_smi)
        except:
            raise ValueError(f"Invalid SMILES: {smi}")

    return canon_smis


def weighted_moment(
    x: ArrayLike, center: Optional[float] = None, n: int = 2, weights: Optional[ArrayLike] = None
) -> float:
    """calculate the n-th moment with given sample weights

    Parameters
    ----------
    x : array_like
        samples
    c : Optional[float], default=None
        central location. If None, the average of `x` is used
    n : int, default=2
        the moment to calculate
    w : Optional[ArrayLike], default=None
        sample weights, if weighted moment is to be computed.

    Returns
    -------
    float
        the n-th moment
    """
    x = np.array(x)
    center = center or np.average(x, weights=weights)

    return np.average((x - center) ** n, weights=weights)


def coarsened_sd(y: np.ndarray, Z: np.ndarray, t: float) -> float:
    """the coarsened standard deviation of the samples `y`

    The coarsened moment is calculated via clustering the input samples `y` according to the input
    linkage matrix `Z` and distance threhsold `t` and calculating the mean value of each cluster.
    The 2nd weighted moment (variance) of these cluster means is calculated with weights equal to
    the size of the respective cluster.

    NOTE: The samples are assumed to lie in the range [0, 1], so the coarsened standard deviation
    is multiplied by 2 to normalize it to the range [0, 1].

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
    """
    if (y < 0).any() or (y > 1).any():
        warnings.warn("Input array 'y' has values outside of [0, 1]")

    clusters = fcluster(Z, t, "distance")

    # get the variance/std dev of the property across clusters
    # we use weights to reduce the size of the `means` array
    means = []
    weights = []
    for i in set(clusters):
        mask = clusters == i
        means.append(y[mask].mean())
        weights.append(len(y[mask]))

    # max std dev is 0.5 --> multiply by 2 so that results is in [0,1]
    var = weighted_moment(means, n=2, weights=weights)
    sd_normalized = 2 * var ** (1 / 2)

    return sd_normalized


def coarse_grain(D: np.ndarray, y: np.ndarray, min_dt: float = 0) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Clustering...")

    Z = max_linkage(D)
    all_distance_thresholds = Z[:, 2]

    # subsample distance_thresholds to avoid doing too many computations on
    # distances that are virtually the same
    thresholds = []
    t_prev = -1
    for t in all_distance_thresholds:
        if t < t_prev + min_dt:
            continue

        thresholds.append(t)
        t_prev = t

    sds = [coarsened_sd(y, Z, t) for t in thresholds]

    # when num_clusters == num_data ==> stddev/skewness of dataset
    thresholds = np.array([0.0, *thresholds, 1.0])
    sds = np.array([coarsened_sd(y, Z, t=-0.1), *sds, coarsened_sd(y, Z, t=1.1)])

    return thresholds, sds


def rogi(
    y: ArrayLike,
    normalize: bool = True,
    X: Optional[np.ndarray] = None,
    fps: Optional[Iterable[ExplicitBitVect]] = None,
    smis: Optional[Iterable[str]] = None,
    metric: Union[str, Metric] = Metric.TANIMOTO,
    max_dist: Optional[float] = None,
    min_dt: float = 0.01,
    nboots: int = 1,
):
    y = np.array(y)
    if normalize:
        y = safe_normalize(y)
    elif (y < 0).any() or (y > 1).any():
        warnings.warn("Input array 'y' has values outside of [0, 1]")

    if X is not None:
        D = calc_distance_matrix_X(X, metric, max_dist)
    elif fps is not None:
        D = calc_distance_matrix_fps(fps, Metric.TANIMOTO)
    elif smis is not None:
        smis = validate_smis(smis)
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        fps = calc_fps(mols, Fingerprint.MORGAN, 2, 2048)
        D = calc_distance_matrix_fps(fps, Metric.TANIMOTO)

    thresholds, sds = coarse_grain(D, min_dt)
    score = sds[0] - trapezoid(sds, thresholds)

    if nboots > 1:
        D_square = squareform(D)
        size = D_square.shape[0]

        boot_scores = []
        for _ in range(nboots):
            idxs = np.random.choice(range(size), size=size, replace=True)
            D = unsquareform(D_square[np.ix_(idxs, idxs)])

            thresholds, sds = coarse_grain(D, min_dt)
            boot_score = sds[0] - trapezoid(sds, thresholds)
            boot_scores.append(boot_score)

        uncertainty = np.std(boot_scores)
    else:
        uncertainty = None

    return score, uncertainty
