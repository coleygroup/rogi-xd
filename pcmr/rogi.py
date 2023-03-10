from itertools import chain
import logging
from typing import Iterable, Optional, Sequence, Union
import warnings

from fastcluster import complete as max_linkage
import numpy as np
from numpy.typing import ArrayLike
from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs import ExplicitBitVect, BulkTanimotoSimilarity, BulkDiceSimilarity
from scipy.cluster.hierarchy import fcluster
from scipy.integrate import trapezoid
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler

from pcmr.utils import Fingerprint, FingerprintConfig, flist, Metric, RogiResult, IntegrationDomain

Input = Union[np.ndarray, Sequence[ExplicitBitVect], Sequence[str]]

logger = logging.getLogger(__name__)


def mask_inputs(xs: Input, y: np.ndarray):
    y_mask = np.isfinite(y)

    if isinstance(xs, np.ndarray):
        x_mask = np.isfinite(xs).all(1)
        mask = y_mask & x_mask
        xs = xs[mask]
    elif isinstance(xs, Sequence) and isinstance(xs[0], ExplicitBitVect):
        mask = y_mask
        xs = [fp for fp, m in zip(xs, mask) if m]
    elif isinstance(xs, Sequence) and isinstance(xs[0], str):
        x_mask = np.array([Chem.MolFromSmiles(smi) is not None for smi in xs], bool)
        mask = y_mask & x_mask
        xs = [smi for smi, m in zip(xs, mask) if m]
    else:
        raise TypeError(
            "arg 'xs' must be of type `np.ndarray` | `Sequence[ExplicitBitVect]` | Sequence[str]!"
        )
    
    return xs, y[mask]


def unsquareform(A: np.ndarray) -> np.ndarray:
    """Return the condensed distance matrix (upper triangular) of the square distance matrix `A`"""
    return A[np.triu_indices(A.shape[0], k=1)]


def estimate_max_dist(X: np.ndarray, metric: Metric) -> float:
    bounds = np.stack([X.min(0), X.max(0)])
    range_ = bounds[1] - bounds[0]

    if metric == Metric.EUCLIDEAN:
        d_max = np.sqrt(np.sum(range_**2))
    elif metric == Metric.CITYBLOCK:
        d_max = np.sum(range_)
    elif metric == Metric.COSINE:
        # in scipy pdist, 1-cosine is returned so that \in [0,2]
        d_max = 2.0
    elif metric == Metric.MAHALANOBIS:
        # more general approach that would work for other metrics too excluding e.g. cosine
        CV = np.atleast_2d(np.cov(X.astype(np.double, copy=False).T))
        VI = np.linalg.inv(CV).T.copy()
        d_max = pdist(bounds, metric="mahalanobis", VI=VI)[0]
    else:
        raise ValueError(f"Invalid metric! got: {metric.value}")

    logger.debug(f"Estimated max dist: {d_max:0.3f}")
    return d_max


def calc_distance_matrix_ndarray(
    X: np.ndarray, metric: Metric, d_max: Optional[float] = None
) -> np.ndarray:
    """Calculate the distance matrix of the input array X

    Raises
    ------
    ValueError
        if X has any invalid input values (e.g., 'inf' or 'nan')
    """
    if not np.isfinite(X).all():
        raise ValueError("arg: 'X' must have only finite vaues!")

    if metric == Metric.PRECOMPUTED:
        logging.info("Using precomputed distance matrix")
        if X.ndim == 1:
            D = X
        elif X.ndim == 2:
            D = unsquareform(X)
        else:
            raise ValueError(f"Precomputed distance matrix must have rank {{1, 2}}. got: {X.ndim}")

        d_max_ = d_max or 1.0
    else:
        D: np.ndarray = pdist(X, metric.value)
        d_max_ = d_max or estimate_max_dist(X, metric)

    # Scaling distances only normalizes the integration domain
    D = D / d_max_

    if (D > 1.0).any():
        raise ValueError(
            "Pairwise distance matrix is not normalized! "
            f"Please ensure the provided 'd_max' is correct. got: {d_max:0.3f}"
        )

    return D


def calc_fps(
    mols: Iterable[Chem.Mol],
    fp: Fingerprint = Fingerprint.MORGAN,
    radius: int = 2,
    length: int = 2048,
) -> list[ExplicitBitVect]:
    logger.info("Calculating fingerprints")
    if fp == Fingerprint.MORGAN:
        fps = [Chem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=length) for m in mols]
    elif fp == Fingerprint.TOPOLOGICAL:
        fps = [Chem.RDKFingerprint(m) for m in mols]
    else:
        raise ValueError(f"Invalid fingerprint! got: {fp.value}")

    return fps


def calc_distance_matrix_fps(
    fps: Sequence[ExplicitBitVect], metric: Metric = Metric.TANIMOTO
) -> np.ndarray:
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


def validate_and_canonicalize_smis(smis: Iterable[str]) -> list[str]:
    canon_smis = []
    for smi in smis:
        try:
            c_smi = Chem.CanonSmiles(smi)
            canon_smis.append(c_smi)
        except:  # noqa: E722
            raise ValueError(f"Invalid SMILES: {smi}")

    return canon_smis


def calc_distance_matrix(
    xs: Input,
    metric: Union[str, Metric, None],
    fp_config: FingerprintConfig = FingerprintConfig(),
    max_dist: Optional[float] = None,
):
    """Calculate the distance matrix of the input molecules

    NOTE: see :func:`~pcmr.rogi.rogi` for details on the following arguments: `xs`, `metric`,
    fp_config`, and `max_dist`

    Parameters
    ----------
    xs: Input
    metric : Union[str, Metric, None], default=None
    fp_config: FingerprintConfig, default=FingerprintConfig()
    max_dist: Optional[float] = None

    Returns
    -------
    np.ndarray
        the upper triangular of the distance matrix as a 1-d vector
    """
    if isinstance(xs, np.ndarray):
        metric = Metric.get(metric) if metric is not None else Metric.EUCLIDEAN
        D = calc_distance_matrix_ndarray(xs, metric, max_dist)
    elif isinstance(xs, Sequence) and isinstance(xs[0], ExplicitBitVect):
        metric = Metric.get(metric) if metric is not None else Metric.TANIMOTO
        D = calc_distance_matrix_fps(xs, metric)
    elif isinstance(xs, Sequence) and isinstance(xs[0], str):
        smis = validate_and_canonicalize_smis(xs)
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        fps = calc_fps(mols, **fp_config._asdict())
        metric = Metric.get(metric) if metric is not None else Metric.TANIMOTO
        D = calc_distance_matrix_fps(fps, Metric.TANIMOTO)
    else:
        raise TypeError(
            "arg 'xs' must be of type `np.ndarray` | `Sequence[ExplicitBitVect]` | `Sequence[str]!`"
        )

    return D


def nmoment(
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
    center = center if center is not None else np.average(x, weights=weights)

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

    cluster_ids = fcluster(Z, t, "distance")
    clusters = set(cluster_ids)

    means = []
    weights = []
    for i in clusters:
        mask = cluster_ids == i
        means.append(y[mask].mean())
        weights.append(len(y[mask]))

    # max std dev is 0.5 --> multiply by 2 so that results is in [0,1]
    var = nmoment(means, n=2, weights=weights)
    sd_normalized = 2 * np.sqrt(var)

    return sd_normalized, len(clusters)


def coarse_grain(
    D: np.ndarray, y: np.ndarray, min_dt: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.debug("Clustering...")
    Z = max_linkage(D)
    all_distance_thresholds = Z[:, 2]

    logger.debug(f"Subsampling with minimum step size of {min_dt:0.3f}")
    thresholds = []
    t_prev = -1
    for t in all_distance_thresholds:
        if t < t_prev + min_dt:
            continue

        thresholds.append(t)
        t_prev = t

    logger.debug(f"Coarsening with thresholds {flist(thresholds):0.3f}")
    cg_sds, n_clusters = zip(*[coarsened_sd(y, Z, t) for t in thresholds])

    # when num_clusters == num_data ==> stddev/skewness of dataset
    thresholds = np.array([0.0, *thresholds, 1.0])
    cg_sds = np.array([2 * y.std(), *cg_sds, 0])
    n_clusters = np.array([len(y), *n_clusters, 1])

    return thresholds, cg_sds, n_clusters


def rogi(
    xs: Input,
    y: ArrayLike,
    normalize: bool = True,
    metric: Union[str, Metric, None] = None,
    fp_config: FingerprintConfig = FingerprintConfig(),
    max_dist: Optional[float] = None,
    min_dt: float = 0.01,
    domain: IntegrationDomain = IntegrationDomain.LOG_CLUSTER_RATIO,
    nboots: int = 1,
) -> RogiResult:
    """calculate the ROGI of a dataset and (optionally) its uncertainty

    NOTE: invalid scores or inputs will be silently removed before calculating the ROGI

    Parameters
    ----------
    xs: Union[np.ndarray, Sequence[ExplicitBitVect], Sequence[str]]
        the input molecules as one of the following types:
        
        * `np.ndarray`: Either (a) the precalculated input representations as a rank-2 matrix OR (b) the precalculated distance matrix (if using Metric.PRECOMPUTED) as a rank-1 (dense) or rank-2 (square) matrix
        * `Sequence[ExplicitBitVect]`: the precalculated input fingerprints as rdkit :class:`ExplicitBitVect`s. 
        * `Sequence[str]`: the SMILES strings of the input molecules

    y : ArrayLike
        the property values
    normalize : bool, default=True
        whether to normalize the property values to the range [0, 1],
    metric : Union[str, Metric, None], default=None
        the distance metric to use or its string alias. If `None`, will choose an appropriate
        distance metric based on the representation supplied:

            1) `np.ndarray`: `Metric.EUCLIDEAN`
            2) `Sequence[ExplicitBitVect]`: `Metric.TANIMOTO`
            3) `Sequence[str]`: `Metric.TANIMOTO`

    fp_config: FingerprintConfig, default=FingerprintConfig()
        the config to use for calculating fingerprints of the input SMILES strings, if necessary.
        See :class:`~pcmr.utils.FingerprintConfig` for more details
    min_dt : float, default=0.01
        the mimimum distance to use between threshold values when coarse graining the dataset,
    log_ratio : bool, default=True
        TODO
    nboots : int, default=1
        the number of samples to use when calculating uncertainty via bootstrapping.
        If `nboots <= 1`, no bootstrapping will be performed

    Returns
    -------
    RogiResult
        the result of the calculation. See :class:`~pcmr.utils.rogi.RogiResult` for more details
    """
    y = np.array(y)

    if normalize:
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1))[:, 0]
    elif (y < 0).any() or (y > 1).any():
        warnings.warn("Input array 'y' has values outside [0, 1]. ROGI may be outside [0, 1]!")

    xs, y_ = mask_inputs(xs, y)
    D = calc_distance_matrix(xs, metric, fp_config, max_dist)

    if (n_invalid := len(y) - len(y_)) > 0:
        logger.info(f"Removed {n_invalid} input(s) with invalid features or scores")

    thresholds, cg_sds, n_clusters = coarse_grain(D, y_, min_dt)
    if domain == IntegrationDomain.THRESHOLD:
        x = thresholds
    elif domain == IntegrationDomain.CLUSTER_RATIO:
        x = 1 - n_clusters / n_clusters[0]
    else:
        x = 1 - np.log(n_clusters) / np.log(n_clusters[0])

    score: float = cg_sds[0] - trapezoid(cg_sds, x)

    if nboots > 1:
        logger.debug(f"Bootstrapping with {nboots} samples")
        D_square = squareform(D)
        size = D_square.shape[0]

        boot_scores = []
        for _ in range(nboots):
            idxs = np.random.choice(range(size), size, True)
            D = unsquareform(D_square[np.ix_(idxs, idxs)])

            thresholds_, cg_sds_, n_clusters_ = coarse_grain(D, y_, min_dt)
            if domain == IntegrationDomain.THRESHOLD:
                x = thresholds_
            elif domain == IntegrationDomain.CLUSTER_RATIO:
                x = 1 - n_clusters_ / n_clusters_[0]
            else:
                x = 1 - np.log(n_clusters_) / np.log(n_clusters_[0])
            boot_score = cg_sds[0] - trapezoid(cg_sds_, x)
            boot_scores.append(boot_score)

        uncertainty = np.std(boot_scores)
    else:
        uncertainty = None

    return RogiResult(score, uncertainty, len(y_), thresholds, cg_sds, n_clusters)
