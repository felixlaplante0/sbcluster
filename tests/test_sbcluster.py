"""Tests for the sbcluster package."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from sbcluster import SpectralBridges, ngap_scorer, silhouette_scorer


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-1.2, -1.0],
            [-1.0, -0.8],
            [-0.8, -1.2],
            [0.9, 1.1],
            [1.1, 0.8],
            [1.3, 1.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    return X, y


def test_fit_predict():
    """Exercises the public fit and predict workflow."""
    X, _ = _data()
    n_clusters = 2
    estimator = SpectralBridges(
        n_clusters=n_clusters,
        n_nodes=4,
        n_iter=5,
        random_state=0,
    )

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    assert clone(estimator).get_params()["n_clusters"] == n_clusters
    assert estimator.fit(X) is estimator

    labels = estimator.predict(X)

    assert labels.shape == (X.shape[0],)
    assert estimator.labels_.shape == (X.shape[0],)
    assert estimator.subcluster_centers_.shape == (4, 2)
    assert estimator.subcluster_labels_.shape == (4,)
    assert estimator.affinity_matrix_.shape == (4, 4)
    assert np.isfinite(estimator.ngap_)

    with pytest.raises(ValueError, match="X has 3 features"):
        estimator.predict(np.column_stack([X, np.ones(X.shape[0])]))


def test_fit_validation():
    """Checks invalid graph-size and sample-size inputs."""
    X, _ = _data()

    with pytest.raises(ValueError, match="n_nodes must be greater"):
        SpectralBridges(n_clusters=3, n_nodes=3).fit(X)

    with pytest.raises(ValueError, match="n_samples=6 must be >= n_nodes=7"):
        SpectralBridges(n_clusters=2, n_nodes=7).fit(X)


@pytest.mark.parametrize("p", [2.0, np.inf])
def test_affinity_matrix(p):
    """Checks finite and infinite affinity aggregation paths."""
    X, _ = _data()
    centers = np.array([[-1.0, -1.0], [1.1, 1.0]])
    labels = np.array([0, 0, 0, 1, 1, 1])

    affinity = SpectralBridges._compute_affinity_matrix(X, centers, labels, p)

    assert affinity.shape == (2, 2)
    assert np.allclose(affinity, affinity.T)
    assert np.isfinite(affinity).all()


def test_scale_and_laplacian():
    """Checks affinity scaling and Laplacian eigendecomposition."""
    affinity = np.array(
        [
            [1.0, 0.7, 0.2],
            [0.7, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ]
    )

    scaled = SpectralBridges._scale_affinity_matrix(
        affinity,
        perplexity=2.0,
        max_iter=4,
    )
    eigvals, eigvecs = SpectralBridges._eigh_laplacian(
        scaled,
        n_components=2,
        tol=1e-8,
    )

    assert scaled.shape == affinity.shape
    assert eigvals.shape == (3,)
    assert eigvecs.shape == (3, 3)
    assert np.isfinite(scaled).all()


def test_scorers():
    """Checks model-selection scorers on fitted and unfitted estimators."""
    X, _ = _data()
    estimator = SpectralBridges(
        n_clusters=2,
        n_nodes=4,
        n_iter=5,
        random_state=0,
    ).fit(X)

    assert np.isfinite(ngap_scorer(estimator))
    assert np.isfinite(silhouette_scorer(estimator, X))

    with pytest.raises(NotFittedError):
        ngap_scorer(SpectralBridges(n_clusters=2, n_nodes=4))
