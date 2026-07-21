"""Normalization utilities for uncertainty quantification."""

import numpy as np


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99), return_confidence=True):
    """
    Quantile power normalize with inversion.
    
    This inverts the values first (1/x) so that higher uncertainty maps to lower rank,
    then applies quantile normalization with power transform.
    
    Args:
        x: Input array of uncertainty values
        gamma: Power transform exponent (default: 0.5)
        clip: Percentile range for clipping (default: (1, 99))
    
    Returns:
        Normalized values in [0, 1] range
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=float)
    finite_values = x[finite]
    epsilon = 1e-9
    
    # Inversion: convert uncertainty to confidence-like score for ranking
    if return_confidence:
        finite_values = 1.0 / (finite_values + epsilon)
    
    if clip is not None:
        lo, hi = np.percentile(finite_values, clip)
        finite_values = np.clip(finite_values, lo, hi)
    
    # Get ranks (ties broken arbitrarily)
    ranks = np.argsort(np.argsort(finite_values)) + 1
    
    # Convert to uniform [0, 1] with power transform
    u = ranks / (len(finite_values) + 1.0)
    result = np.full_like(x, np.nan, dtype=float)
    result[finite] = u ** gamma
    return result
