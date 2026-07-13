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
    epsilon = 1e-9
    
    # Inversion: convert uncertainty to confidence-like score for ranking
    if return_confidence:
        x = 1.0 / (x + epsilon)
    
    if clip is not None and len(x) > 0:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    
    # Get ranks (ties broken arbitrarily)
    ranks = np.argsort(np.argsort(x)) + 1
    
    # Convert to uniform [0, 1] with power transform
    u = ranks / (len(x) + 1.0)
    return u ** gamma
