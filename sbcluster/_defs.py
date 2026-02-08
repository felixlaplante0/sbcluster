from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from ._cluster import SpectralBridges


# Protocols
@runtime_checkable
class AffinityTransform(Protocol):
    """Protocol for affinity transforms.

    Use this protocol to define custom affinity transforms.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


# Transformations
class ExpQuantileTransform(AffinityTransform):
    """Exponential quantile transform.

    Attributes:
        alphas (tuple[float, float]): Quantiles for affinity matrix computation.
        mult_factor (int | float): Scaling parameter for affinity matrix computation.
    """

    q1: float
    q2: float
    mult_factor: float

    @validate_params(
        {
            "q1": [Interval(Real, 0, 1, closed="left")],
            "q2": [Interval(Real, 0, 1, closed="left")],
            "mult_factor": [Interval(Real, 0, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, q1: float, q2: float, mult_factor: float):
        """Initialize the Exponential quantile transform.

        Args:
            q1 (float): First quantile for affinity matrix computation.
            q2 (float): Second quantile for affinity matrix computation.
            mult_factor (float): Scaling parameter for affinity matrix
                computation.
        """
        if not (0 <= q1 < q2 <= 1):
            raise ValueError("q1 and q2 must be between 0 and 1 and q1 < q2")

        self.q1 = q1
        self.q2 = q2
        self.mult_factor = mult_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        q1, q2 = np.quantile(x, (self.q1, self.q2))
        gamma = np.log(self.mult_factor) / (q2 - q1)
        return np.exp(gamma * (x - x.max()))


# Scorers
def silhouette_scorer(estimator: SpectralBridges, X: np.typing.ArrayLike) -> float:
    """Silhouette scorer for SpectralBridges.

    Args:
        estimator (SpectralBridges): The estimator to score.
        X (np.typing.ArrayLike): The data to score.

    Returns:
        float: The silhouette score.
    """
    return silhouette_score(X, estimator.predict(X))


def ngap_scorer(estimator: SpectralBridges, *_args: Any, **_kwargs: Any) -> float:
    """NGAP scorer for SpectralBridges.

    Args:
        estimator (SpectralBridges): The estimator to score.

    Raises:
        ValueError: If the estimator is not fitted.

    Returns:
        float: The NGAP score.
    """
    check_is_fitted(estimator, "ngap_")

    return cast(float, estimator.ngap_)
