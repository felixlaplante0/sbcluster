from numbers import Integral, Real
from typing import Final, Self, cast

import numpy as np
import numpy.typing as npt
from fastkmeanspp import KMeans
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted

from ._defs import AffinityTransform, ExpQuantileTransform

# Constants
DEFAULT_AFFINITY_TRANSFORM: Final[AffinityTransform] = ExpQuantileTransform(
    0.1, 0.9, 1e4
)


def _compute_affinity(
    X: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray, p: int | float
) -> np.ndarray:
    """Compute the affinity matrix.

    Args:
        X (np.ndarray): The data points.
        cluster_centers (np.ndarray): The cluster centers.
        labels (np.ndarray): The labels of the data points.
        p (int | float): The power of the affinity matrix.

    Returns:
        np.ndarray: The affinity matrix.
    """
    X_centered = [
        np.array(
            X[labels == i] - cluster_centers[i],
            dtype=np.float64,
            order="F",
        )
        for i in range(cluster_centers.shape[0])
    ]

    affinity_matrix = np.empty(
        (cluster_centers.shape[0], cluster_centers.shape[0]), dtype=np.float64
    )
    for i in range(cluster_centers.shape[0]):
        segments = np.asfortranarray(
            cluster_centers - cluster_centers[i],
            dtype=np.float64,
        )
        dists = np.einsum("ij,ij->i", segments, segments)
        dists[i] = 1

        projs = cast(np.ndarray, dgemm(1.0, X_centered[i], segments, trans_b=True))
        projs /= dists

        # Numerically stable computation of the affinity matrix
        if p < np.inf:
            log_proj = np.log(np.clip(projs, np.finfo(np.float64).tiny, None))
            m = log_proj.max(axis=0)
            affinity_matrix[i] = p * m + logsumexp(p * (log_proj - m), axis=0)
        else:
            affinity_matrix[i] = np.clip(projs.max(axis=0), 0, None)

    if p < np.inf:
        counts = np.array([e.shape[0] for e in X_centered])
        log_counts = np.log(counts[None, :] + counts[:, None])
        affinity_matrix = np.exp(
            (np.logaddexp(affinity_matrix, affinity_matrix.T) - log_counts) / p
        )
    else:
        affinity_matrix = np.maximum(affinity_matrix, affinity_matrix.T)

    return affinity_matrix


def _eigh_laplacian(
    affinity: np.ndarray,
    n_components: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvectors and eigenvalues of the Laplacian matrix.

    Args:
        affinity (np.ndarray): The affinity matrix.
        n_components (int): The number of components to compute.
        tol (float): The tolerance for the normalized eigengap.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvectors and eigenvalues of the Laplacian
            matrix.
    """
    d = np.power(affinity.mean(axis=1), -0.5)
    L = -(d[:, None] * affinity * d[None, :])
    np.fill_diagonal(L, L.shape[0] + tol)

    return cast(
        tuple[np.ndarray, np.ndarray],
        eigh(
            L,
            subset_by_index=[0, n_components],
        ),
    )


class SpectralBridges(BaseEstimator, ClusterMixin):
    """Spectral Bridges clustering algorithm.

    Attributes:
        n_clusters (int): The number of clusters to form.
        n_nodes (int): Number of nodes or initial clusters.
        p (int | float): Power of the alpha_i.
        n_iter (int): Number of iterations to run the k-means algorithm.
        n_local_trials (int | None): Number of seeding trials for centroids
            initialization.
        random_state (int | None): Determines random number generation for centroid
            initialization.
        tol (float): Tolerance for the normalized eigengap.
        affinity_transform (AffinityTransform): Affinity transform to apply to the
            affinity matrix.
        cluster_centers_ (np.ndarray | None): Coordinates of cluster centers.
        cluster_labels_ (np.ndarray | None): Labels of each cluster.
        labels_ (np.ndarray | None): Labels of each data point.
        affinity_matrix_ (np.ndarray | None): Affinity matrix.
        ngap_ (float | None): The normalized eigengap.
    """

    n_clusters: int
    n_nodes: int
    p: int | float
    n_iter: int
    n_local_trials: int | None
    random_state: int | None
    tol: float
    affinity_transform: AffinityTransform
    cluster_centers_: np.ndarray | None
    cluster_labels_: np.ndarray | None
    labels_: np.ndarray | None
    affinity_matrix_: np.ndarray | None
    ngap_: float | None

    @validate_params(
        {
            "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "n_nodes": [Interval(Integral, 2, None, closed="left")],
            "p": [Interval(Real, 0, None, closed="right")],
            "n_iter": [Interval(Integral, 1, None, closed="left")],
            "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
            "random_state": ["random_state"],
            "tol": [Interval(Real, 0, None, closed="left")],
            "affinity_transform": [AffinityTransform],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_clusters: int,
        n_nodes: int,
        *,
        p: int | float = 2,
        n_iter: int = 20,
        n_local_trials: int | None = None,
        random_state: int | None = None,
        tol: float = 1e-8,
        affinity_transform: AffinityTransform = DEFAULT_AFFINITY_TRANSFORM,
    ):
        """Initialize the Spectral Bridges model.

        Args:
            n_clusters (int): The number of clusters to form.
            n_nodes  (int | None): Number of nodes or initial clusters.
            p (int | float, optional): Power of the alpha_i. Defaults to 2.
            n_iter (int, optional): Number of iterations to run the k-means
                algorithm. Defaults to 20.
            n_local_trials (int | None, optional): Number of seeding trials for
                centroids initialization. Defaults to None.
            random_state (int | None, optional): Determines random number
                generation for centroid initialization. Defaults to None.
            tol (float, optional): Tolerance for the normalized eigengap.
                Defaults to 1e-8.
            affinity_transform (AffinityTransform, optional): Affinity transform
                to apply to the affinity matrix. Defaults to DEFAULT_AFFINITY_TRANSFORM.
        """
        self.n_clusters = n_clusters
        self.n_nodes = n_nodes
        self.p = p
        self.n_iter = n_iter
        self.n_local_trials = n_local_trials
        self.random_state = random_state
        self.tol = tol
        self.affinity_transform = affinity_transform
        self.cluster_centers_ = None
        self.ngap_ = None
        self.affinity_matrix_ = None
        self.eigvals_ = None
        self.eigvecs_ = None

        if self.n_nodes <= self.n_clusters:
            raise ValueError(
                f"n_nodes must be greater than n_clusters, got {self.n_nodes} <= "
                f"{self.n_clusters}"
            )

    @validate_params(
        {
            "X": ["array-like"],
            "y": [None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: npt.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fit the Spectral Bridges model on the input data X.

        Args:
            X (npt.ArrayLike): Input data to cluster.
            y (None, optional): Placeholder for y.

        Raises:
            ValueError: If the number of samples is less than the number of clusters.
            ValueError: If `X` contains inf or NaN values.

        Returns:
            Self: The fitted model.
        """
        X = np.asarray(X)  # type: ignore

        if X.shape[0] < self.n_nodes:
            raise ValueError(
                f"n_samples={X.shape[0]} must be >= n_nodes={self.n_nodes}."
            )

        kmeans = KMeans(
            self.n_nodes,
            self.n_iter,
            self.n_local_trials,
            self.random_state,
        ).fit(X)
        self.cluster_centers_ = cast(np.ndarray, kmeans.cluster_centers_)

        self.affinity_matrix_ = self.affinity_transform(
            _compute_affinity(
                X, self.cluster_centers_, cast(np.ndarray, kmeans.labels_), self.p
            )
        )

        eigvals, eigvecs = _eigh_laplacian(
            self.affinity_matrix_, self.n_clusters, self.tol
        )

        eigvecs = eigvecs[:, :-1]
        eigvecs /= np.linalg.norm(eigvecs, axis=1)[:, None]
        self.ngap_ = (eigvals[-1] - eigvals[-2]) / eigvals[-2]

        self.cluster_labels_ = cast(
            np.ndarray,
            KMeans(self.n_clusters, self.n_iter, self.n_local_trials, self.random_state)
            .fit(eigvecs)
            .labels_,
        )
        self.labels_ = self.cluster_labels_[kmeans.labels_]

        return self

    @validate_params(
        {
            "X": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the nearest cluster index for each input data point x.

        Args:
            X (npt.ArrayLike): The input data.

        Raises:
            ValueError: If `X` contains inf or NaN values.
            ValueError: If `self.cluster_centers_` and `self.cluster_labels_` are not
                set.

        Returns:
            np.ndarray The predicted cluster indices.
        """
        check_is_fitted(self, ("cluster_centers_", "cluster_labels_"))

        dummy_kmeans = KMeans(
            self.n_clusters,
            self.n_iter,
            self.n_local_trials,
            self.random_state,
        )
        dummy_kmeans.cluster_centers_ = self.cluster_centers_
        dummy_kmeans.labels_ = self.cluster_labels_

        return cast(np.ndarray, self.cluster_labels_)[dummy_kmeans.predict(X)]
