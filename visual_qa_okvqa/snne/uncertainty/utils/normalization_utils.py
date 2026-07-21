"""Normalization utilities for uncertainty quantification."""

import numpy as np


def quantile_power_normalize(x, gamma=0.5, clip=(1, 99), return_confidence=True):
    """Rank-normalize uncertainty, optionally converting it to confidence first."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=float)
    finite_values = x[finite]
    if return_confidence:
        finite_values = 1.0 / (finite_values + 1e-9)
    if clip is not None:
        lower, upper = np.percentile(finite_values, clip)
        finite_values = np.clip(finite_values, lower, upper)
    ranks = np.argsort(np.argsort(finite_values)) + 1
    result = np.full_like(x, np.nan, dtype=float)
    result[finite] = (ranks / (len(finite_values) + 1.0)) ** gamma
    return result
