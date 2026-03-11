from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from ._cluster import SpectralBridges


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
