"""
module for calculating the ROuGhness Index (ROGI), heavily borrowed from
https://github.com/coleygroup/rogi/blob/main/src/rogi/roughness_index.py
"""
from typing import Iterable, Tuple, Optional, Union
import warnings

from fastcluster import complete as MaxLinkage
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform, pdist

from pcmr.utils import Fingerprint, Metric


def _safe_normalize(x: ArrayLike, eps: float = 1e-6):
    x = np.array(x)
    if x.max() - x.min() > eps:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return x - x.min()


def unsquareform(a: np.ndarray):
    return a[np.triu_indices(a.shape[0], k=1)]


def nmoment(x: ArrayLike, c: Optional[float] = None, n: int = 2, w: Optional[ArrayLike] = None):
    """Estimate the n-th moment.

    Parameters
    ----------
    x : array_like
        Samples.
    c : Optional[float], default=None
        Central location. If None, the average of `x` is used
    n : int, default=2
        the moment to calculate
    w : Optional[ArrayLike], default=None
        sample weights, if weighted moment is to be computed.

    Returns
    -------
        float
            Value of the n-th moment.
    """
    x = np.array(x)
    c = c or np.average(x, weights=w)

    return np.average((x - c) ** n, weights=w)


class RoughnessIndex:
    def __init__(
        self,
        Y: ArrayLike,
        norm_Y: bool = True,
        smiles: Optional[ArrayLike] = None,
        fps: Optional[ArrayLike] = None,
        X: Optional[NDArray] = None,
        max_dist: Optional[float] = None,
        metric: Union[str, Metric] = Metric.TANIMOTO,
        verbose: bool = True,
    ):
        """A measure of roughness for molecular property datasets.

        Parameters
        ----------
        Y : ArrayLike
            List of property values, either continuous or binary.
        norm_Y : bool, default=True
            Whether to normalize the property (Y) values. Note that Y values need to be between
            zero and one for the, ROGI score to be bounded by one.
        smiles : Optional[ArrayLike], default=None
            Array/List of SMILES strings for the molecules with properties Y. SMILES need to be
            provided if `fps` or `X` is not provided.
        fps : Optional[ArrayLike], default=None
            2d array with pre-computed fingerprints for the molecules with properties Y
        X : Optional[NDArray], default=None
            2d array with descriptors for the molecules with properties Y. If the `metric` chosen
            is `precomputed`, then provide a square distance matrix here. Default is None.,
        max_dist : Optional[float], default=None
            Maximum distance achievable between any two molecules given the chosen descriptors `X`
            and metric. If `None`, this is estimated from the dataset.
        metric : str, default="tanimoto"
            Distance metric to use. Available metrics are: "tanimoto", "euclidean", "cityblock"
            "cosine", "mahalanobis", and "precomputed". When using fingerprints, "tanimoto" is used
            by default. With descriptors, any other distance may be used. If choosing
            "precomputed", please provide a square distance matrix for the argument `X`.,
        verbose : bool, default=True
            Whether to print some information to screen. Default is True.,

        Raises
        ------
        ValueError
            if `smis`, `fps`, and `X`, are all `None`
        """
        self.norm_Y = norm_Y
        self.Y = Y
        self.smiles = smiles
        self.fps = fps
        self.X = X
        self.max_dist = max_dist
        self.metric = metric if isinstance(metric, Metric) else Metric.get(metric)
        self.verbose = verbose

        self.validate_inputs()

        self.Z = None
        self.distance_thresholds = None
        self.property_moments = None
        self.auc = None

        # normalize property values
        if self.norm_Y is True:
            self._Y = _safe_normalize(Y)
        else:
            self._Y = np.array(Y)

        # check we have at least one input for the domain
        if self.smiles is None and self.fps is None and self.X is None:
            raise ValueError("No smiles, fingerprint, or descriptor input found")

        # compute fingerprints and/or distance matrix if needed
        self._fps = fps
        self._Dx = self._parse_X_input()

        # if self._Dx is None, no descriptors were provided and we compute fingerprints
        if self._Dx is None:
            if self._fps is None:
                self._smiles = self.validate_smis(smiles)
                self._mols = [Chem.MolFromSmiles(smi) for smi in self._smiles]
                self._fps = self.compute_fingerprints(
                    self._mols, Fingerprint.MORGAN, 2, 2048, self.verbose
                )

            self._Dx = self._compute_distance_matrix(self._fps, Metric.TANIMOTO, self.verbose)

    def validate_inputs(self):
        # when Y_norm is False, we still expect Y \in [0,1], i.e. pre-normalized
        if self.norm_Y is False:
            if any(np.array(self.Y) > 1) or any(np.array(self.Y) < 0):
                if self.verbose is True:
                    warnings.warn(
                        "all property values of Y are expected to be between 0 and 1."
                        "Roughness Index will not be normalized and may exceed 1."
                    )

        # if descriptors provided, make sure correct format
        if self.X is not None:
            if isinstance(self.X, pd.DataFrame):
                self._X = self.X.to_numpy()
            else:
                self._X = np.array(self.X)

            if self.metric == Metric.PRECOMPUTED:
                if self._X.ndim not in [1, 2]:
                    raise ValueError(f"X expected to be a 1-d or 2-d array")
            else:
                if self._X.ndim != 2:
                    raise ValueError(f"X expected to be a 2-d array or pandas DataFrame")
        else:
            self._X = None

    def _parse_X_input(self):
        """Parse input descriptors to get distance matrix"""
        if self._X is None:
            return None

        # if precomputed, it should be a distance matrix
        if self.metric == Metric.PRECOMPUTED:
            # convert to pdist
            if self._X.ndim == 2:
                Dx = unsquareform(self._X)
            else:
                Dx = self._X

            if self.max_dist is None:
                self._max_dist = 1.0  # i.e. assume already normalized
            else:
                # if max possible distance is provided, use that
                self._max_dist = self.max_dist
        else:
            # compute distance matrix using original X space
            # which will be used for clustering
            Dx = pdist(self._X, metric=self.metric)

            # estimate max distance from range of descriptors provided if max_dist not given
            if self.max_dist is None:
                num_dim = self._X.shape[1]
                ranges = np.array(
                    [self._X[:, j].max() - self._X[:, j].min() for j in range(num_dim)]
                )
                if self.metric == Metric.EUCLIDEAN:
                    self._max_dist = np.sqrt(np.sum(ranges**2))
                elif self.metric == Metric.CITYBLOCK:
                    self._max_dist = np.sum(ranges)
                elif self.metric == Metric.COSINE:
                    self._max_dist = 2.0  # in scipy pdist, 1-cosine is returned so that \in [0,2]
                elif self.metric == Metric.MAHALANOBIS:
                    # more general approach that would work for other metrics too
                    # excluding e.g. cosine
                    Xcorners = np.stack([np.min(self._X, axis=0), np.max(self._X, axis=0)])
                    CV = np.atleast_2d(np.cov(self._X.astype(np.double, copy=False).T))
                    VI = np.linalg.inv(CV).T.copy()
                    self._max_dist = pdist(Xcorners, metric="mahalanobis", VI=VI)[0]
                else:
                    raise ValueError(f"Invalid metric! got: {self.metric}")
            else:
                # if max possible distance is provided, use that
                self._max_dist = self.max_dist

                # normalize distances based on largest possible distance
        # scaling distances does not change clustering results
        # it only normalizes the domain of the integration
        Dx = Dx / self._max_dist

        # check that we Dx \in [0,1]
        if any(Dx > 1.0):
            raise ValueError(
                "Pairwise distance matrix is not normalized. "
                "Make sure the provided maximum distance is correct."
            )

        return Dx

    @property
    def Y(self) -> np.ndarray:
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        if self.norm_Y is True:
            self.__Y = _safe_normalize(Y)
        else:
            self.__Y = np.array(Y)

    @staticmethod
    def validate_smis(smis: str):
        canon_smis = []
        for smi in smis:
            try:
                c_smi = Chem.CanonSmiles(smi)
                canon_smis.append(c_smi)
            except:
                raise ValueError("Invalid SMILES:", smi)

        return canon_smis

    @staticmethod
    def compute_fingerprints(
        mols: Iterable[Chem.Mol],
        fp: Fingerprint = Fingerprint.MORGAN,
        radius: int = 2,
        nBits: int = 2048,
        verbose: bool = False,
    ):
        if verbose is True:
            print("Computing fingerprints...")

        if fp == Fingerprint.TOPOLOGICAL:
            fps = [Chem.RDKFingerprint(m) for m in mols]
        elif fp == Fingerprint.MORGAN:
            fps = [
                AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits) for m in mols
            ]
        else:
            raise ValueError(f"fingerprint {fp} not implemented")

        return fps

    @staticmethod
    def _compute_distance_matrix(fps, metric: Metric = Metric.TANIMOTO, verbose: bool = False):
        # metrics: Tanimoto, Dice
        if verbose is True:
            print("Computing distance matrix...")

        if metric == Metric.TANIMOTO:
            sim_func = DataStructs.BulkTanimotoSimilarity
        elif metric == Metric.DICE:
            sim_func = DataStructs.BulkDiceSimilarity
        else:
            raise ValueError(f"metric {metric} not implemented")

        # we compute a condensed distance matrix, i.e. upper triangular as 1D array
        # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
        num_fps = len(fps)
        Dx = []
        for i in range(num_fps):
            # i+1 becauase we know the diagonal is zero
            sim = np.array(sim_func(fps[i], fps[i + 1 :]))
            dist = 1.0 - sim
            Dx.extend(list(dist))
        return np.array(Dx)

    def _get_second_moments(self, t):
        clusters = fcluster(self.Z, t=t, criterion="distance")
        # get the variance/std dev of the property across clusters
        # we use weights to reduce the size of the ``means`` array
        means = []
        weights = []
        for i in set(clusters):
            mask = clusters == i
            m = np.mean(self._Y[mask])
            w = len(self._Y[mask])
            means.append(m)
            weights.append(w)

        variance = nmoment(means, c=None, n=2, w=weights)
        # return normalized second moment
        # max std dev is 0.5 ==> multiply by 2 so that results is [0,1]
        norm_stddev = 2 * np.sqrt(variance)
        return norm_stddev

    def _compute_distances_and_moments(self, Dx, min_dt=0):
        if self.verbose is True:
            print("Clustering...")

        # compute linkage matrix
        self.Z = MaxLinkage(Dx)
        all_distance_thresholds = self.Z[:, 2]

        # subsample distance_thresholds to avoid doing too many computations on
        # distances that are virtually the same
        distance_thresholds = []
        t_prev = -1
        for t in all_distance_thresholds:
            if t - min_dt < t_prev:
                continue
            distance_thresholds.append(t)
            t_prev = t

        moments = []
        for t in distance_thresholds:
            m = self._get_second_moments(t)
            moments.append(m)

        # ensure we have the endpoints, i.e. t=0 and t=1
        distance_thresholds = np.array([0.0] + list(distance_thresholds) + [1.0])
        # when num_clusters = num_data --> stddev/skewness of dataset
        moments = np.array(
            [self._get_second_moments(t=-0.1)] + moments + [self._get_second_moments(t=1.1)]
        )

        return distance_thresholds, moments

    def compute(self, min_dt: float = 0.01, nboots: int = 1) -> Union[float, Tuple[float, float]]:
        """Computes the ROGI score.

        Parameters
        ----------
        min_dt : float
            Smallest dt allowed for determining the thresholds t used for clustering and for numerical integration.
            Default is 0.01.
        nboots : int
            Number of bootstrap samples used to estimate the standard error of ROGI. If 1, the uncertainty is not
            estimated. Default is 1.

        Returns
        -------
        tuple
            ROGI score and its uncertainty. The uncertainty is `None` if `nboots` is less than 2.
        """

        # cluster and compute distances vs variance
        distance_thresholds, property_moments = self._compute_distances_and_moments(
            Dx=self._Dx, min_dt=min_dt
        )

        # threshold used for clustering (from 0 to 1)
        self.distance_thresholds = distance_thresholds
        # decreasing dispersion/skewness with larger clusters
        self.property_moments = property_moments

        # integrate by trapeziodal rule
        dx = np.diff(self.distance_thresholds)
        fx = (self.property_moments[:-1] + self.property_moments[1:]) * 0.5
        self.auc = np.dot(fx, dx)

        rogi_score = self.property_moments[0] - self.auc

        # compute uncertainty
        if nboots > 1:
            _squared_Dx = squareform(self._Dx)
            size = _squared_Dx.shape[0]
            boot_scores = []
            for nboot in range(nboots):
                # get bootstrap indixes
                boot_idx = np.random.choice(range(size), size=size, replace=True)

                # subsample distance matrix
                Dx = unsquareform(_squared_Dx[np.ix_(boot_idx, boot_idx)])

                # cluster and compute distances vs variance
                distance_thresholds, property_moments = self._compute_distances_and_moments(
                    Dx=Dx, min_dt=min_dt
                )

                # integrate by trapeziodal rule
                dx = np.diff(distance_thresholds)
                fx = (property_moments[:-1] + property_moments[1:]) * 0.5
                auc = np.dot(fx, dx)

                boot_score = property_moments[0] - auc
                boot_scores.append(boot_score)

            return rogi_score, np.std(boot_scores)
        else:
            return rogi_score
