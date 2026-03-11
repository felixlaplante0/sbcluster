from numbers import Integral, Real
from typing import Self, cast

import numpy as np
from fastkmeanspp import KMeans
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore


class SpectralBridges(ClusterMixin, BaseEstimator):
    r"""Spectral Bridges clustering algorithm.

    This estimator performs clustering by constructing an affinity matrix from
    the input data and analyzing the spectral structure of the resulting graph.
    The algorithm relies on the eigengap of the normalized affinity matrix to
    detect cluster separation, followed by a k-means step in the spectral
    embedding space.

    The affinity matrix is built using a perplexity-based neighborhood scheme,
    and may optionally be raised to a power :math:`p` to emphasize indirect
    connections between nodes. Clustering is then performed on the leading
    eigenvectors of the normalized graph Laplacian.

    Clusters are obtained by applying k-means to the spectral embedding. The
    normalized eigengap is used as a stability diagnostic for the clustering
    structure.

    Clustering settings:
        - `n_clusters`: Number of clusters to form.
        - `n_nodes`: Number of nodes or initial graph vertices used to construct
          the affinity matrix.

    Affinity construction settings:
        - `perplexity`: Target perplexity used to adapt local neighborhood
          bandwidths when building the affinity matrix.
        - `p`: Power applied to the affinity matrix to emphasize higher-order
          connectivity.

    k-means settings:
        - `n_iter`: Maximum number of iterations for the k-means algorithm.
        - `n_local_trials`: Number of seeding trials used during centroid
          initialization.
        - `random_state`: Controls random initialization of centroids.

    Convergence and diagnostics:
        - `tol`: Tolerance threshold for the normalized eigengap used as a
          diagnostic for cluster separation.

    Attributes:
        n_clusters (int): Number of clusters requested by the user.
        n_nodes (int): Number of graph nodes used to construct the affinity
            structure.
        p (float): Power applied to the affinity matrix.
        perplexity (float): Target perplexity used when computing affinities.
        n_iter (int): Maximum number of iterations for the k-means step.
        n_local_trials (int | None): Number of centroid initialization trials.
        random_state (int | None): Random state used for reproducibility.
        tol (float): Tolerance threshold for the normalized eigengap.
        cluster_centers_ (np.ndarray | None): Coordinates of the cluster
            centers in the spectral embedding.
        cluster_labels_ (np.ndarray | None): Labels assigned to the clusters.
        labels_ (np.ndarray | None): Cluster label assigned to each data point.
        affinity_matrix_ (np.ndarray | None): Computed affinity matrix.
        ngap_ (float | None): Normalized eigengap of the affinity matrix.

    Examples:
        >>> model = SpectralBridges(n_clusters=3, n_nodes=50, perplexity=30)
        >>> labels = model.fit_predict(X)
    """

    n_clusters: int
    n_nodes: int
    p: float
    perplexity: float
    n_iter: int
    n_local_trials: int | None
    random_state: int | None
    tol: float
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
            "perplexity": [Interval(Real, 1, None, closed="neither")],
            "n_iter": [Interval(Integral, 1, None, closed="left")],
            "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
            "random_state": ["random_state"],
            "tol": [Interval(Real, 0, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_clusters: int,
        n_nodes: int,
        *,
        p: float = 2.0,
        perplexity: float = 2.0,
        n_iter: int = 20,
        n_local_trials: int | None = None,
        random_state: int | None = None,
        tol: float = 1e-8,
    ):
        """Initialize the Spectral Bridges model.

        Args:
            n_clusters (int): The number of clusters to form.
            n_nodes  (int | None): Number of nodes or initial clusters.
            p (float, optional): Power applied to the affinity matrix. Defaults to 2.0.
            perplexity (float, optional): Target perplexity for the affinity matrix.
                Defaults to 2.0.
            n_iter (int, optional): Number of iterations to run the k-means
                algorithm. Defaults to 20.
            n_local_trials (int | None, optional): Number of seeding trials for
                centroids initialization. Defaults to None.
            random_state (int | None, optional): Determines random number
                generation for centroid initialization. Defaults to None.
            tol (float, optional): Tolerance for the normalized eigengap.
                Defaults to 1e-8.
        """
        self.n_clusters = n_clusters
        self.n_nodes = n_nodes
        self.p = p
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.n_local_trials = n_local_trials
        self.random_state = random_state
        self.tol = tol
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

    @staticmethod
    def _compute_affinity_matrix(
        X: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray, p: float
    ) -> np.ndarray:
        """Compute the affinity matrix.

        Args:
            X (np.ndarray): The data points.
            cluster_centers (np.ndarray): The cluster centers.
            labels (np.ndarray): The labels of the data points.
            p (float): The power of the affinity matrix.

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

    @staticmethod
    def _scale_affinity_matrix(
        affinity_matrix: np.ndarray,
        perplexity: float,
        low: float = 0.0,
        high: float = 1000.0,
        max_iter: int = 16,
        tol: float = 1e-2,
    ) -> np.ndarray:
        """Scales the affinity matrix to a target perplexity.

        Args:
            affinity_matrix (np.ndarray): The affinity matrix.
            perplexity (float): The target perplexity.
            low (float, optional): The lower bound for the binary search.
                Defaults to 0.0.
            high (float, optional): The upper bound for the binary search.
                Defaults to 1000.0.
            max_iter (int, optional): The maximum number of iterations for the binary
                search. Defaults to 16.
            tol (float, optional): The relative tolerance for the binary search.
                Defaults to 1e-2.

        Returns:
            np.ndarray: The scaled affinity matrix.
        """

        def _perplexity(gamma: float, A: np.ndarray) -> float:
            log_A = gamma * A
            np.fill_diagonal(log_A, -np.inf)
            log_sum_A = cast(np.ndarray, logsumexp(log_A, axis=1))

            log_P = log_A - log_sum_A
            P = np.exp(log_P)
            np.fill_diagonal(log_P, 0.0)

            entropy = -np.sum(P * log_P, axis=1)
            return np.exp(cast(float, np.mean(entropy)))

        affinity_matrix = affinity_matrix - affinity_matrix.max()
        gamma = (low + high) / 2
        for _ in range(max_iter):
            perp = _perplexity(gamma, affinity_matrix)
            if perp > perplexity:
                low = gamma
            else:
                high = gamma
            gamma = (low + high) / 2
            if np.abs(perp - perplexity) / perplexity < tol:
                break

        return np.exp(gamma * affinity_matrix)

    @staticmethod
    def _eigh_laplacian(
        affinity_matrix: np.ndarray,
        n_components: int,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the eigenvectors and eigenvalues of the Laplacian matrix.

        Args:
            affinity (np.ndarray): The affinity matrix.
            n_components (int): The number of components to compute.
            tol (float): The tolerance for the normalized eigengap.

        Returns:
            tuple[np.ndarray, np.ndarray]: The eigenvectors and eigenvalues of the
                normalized Laplacian matrix.
        """
        d = np.power(affinity_matrix.mean(axis=1), -0.5)
        L = -(d[:, None] * affinity_matrix * d[None, :])
        np.fill_diagonal(L, L.shape[0] + tol)

        return cast(
            tuple[np.ndarray, np.ndarray],
            eigh(
                L,
                subset_by_index=[0, n_components],
            ),
        )

    @validate_params(
        {
            "X": ["array-like"],
            "y": [None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fit the Spectral Bridges model on the input data X.

        Args:
            X (np.typing.ArrayLike): Input data to cluster.
            y (None, optional): Placeholder for y.

        Raises:
            ValueError: If the number of samples is less than the number of clusters.
            ValueError: If `X` contains inf or NaN values.

        Returns:
            Self: The fitted model.
        """
        X = np.asarray(validate_data(self, X))  # type: ignore

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

        affinity_matrix = self._compute_affinity_matrix(
            X, self.cluster_centers_, cast(np.ndarray, kmeans.labels_), self.p
        )
        self.affinity_matrix_ = self._scale_affinity_matrix(
            affinity_matrix, self.perplexity
        )

        eigvals, eigvecs = self._eigh_laplacian(
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
    def predict(self, X: np.typing.ArrayLike) -> np.ndarray:
        """Predict the nearest cluster index for each input data point x.

        Args:
            X (np.typing.ArrayLike): The input data.

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
