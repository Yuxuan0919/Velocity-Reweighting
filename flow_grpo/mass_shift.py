"""Groupwise exponential mass redistribution for the Step 2 experiments."""

import numpy as np


def _parameters(gamma_prime, epsilon):
    gamma_prime, epsilon = float(gamma_prime), float(epsilon)
    if not np.isfinite(gamma_prime) or gamma_prime <= 0:
        raise ValueError("gamma_prime must be finite and strictly positive")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    return gamma_prime, epsilon


def _vector(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return values


def l1_exponential_weights(magnitudes, gamma_prime=1.0, epsilon=1e-6):
    """Return centered exponential weights, gamma, and mean absolute deviation.

    MAD is around the arithmetic mean, not the median. A common exponential
    factor is removed: q_i = exp(gamma * (a_i - max(a))). For an empty side,
    q is empty, MAD is zero, and gamma follows the epsilon-floor convention.
    With gamma_prime=1 and n samples, centered logits lie in [-n, 0].
    """
    gamma_prime, epsilon = _parameters(gamma_prime, epsilon)
    magnitudes = _vector(magnitudes, "magnitudes")
    if np.any(magnitudes < 0):
        raise ValueError("magnitudes must be nonnegative")
    mad = 0.0
    if magnitudes.size:
        # Translation and scaling avoid overflow in the mean and cancellation
        # from a large common offset; undo the scale after computing the MAD.
        span = magnitudes.max() - magnitudes.min()
        if span > 0:
            normalized = (magnitudes - magnitudes.min()) / span
            mad = float(span * np.mean(np.abs(normalized - normalized.mean())))
    gamma = gamma_prime / max(mad, epsilon)
    if not np.isfinite(gamma):
        raise ValueError("effective gamma must be finite")
    if not magnitudes.size:
        return magnitudes.copy(), gamma, mad
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(gamma * (magnitudes - magnitudes.max()))
    if np.any(weights == 0):
        raise ValueError("exponential weights underflow; reduce gamma_prime")
    return weights, gamma, mad


def _transform_positive(result, gamma_prime, epsilon):
    positive = result > 0
    magnitudes = result[positive]
    if magnitudes.size < 2 or np.all(magnitudes == magnitudes[0]):
        return
    with np.errstate(over="ignore"):
        mass = magnitudes.sum()
    if not np.isfinite(mass):
        raise ValueError("total positive mass must be finite")
    weights, _, _ = l1_exponential_weights(magnitudes, gamma_prime, epsilon)
    result[positive] = mass * (weights / weights.sum())


def exponential_positive_shifts(shifts, gamma_prime=1.0, epsilon=1e-6):
    """Scheme 3: redistribute positive mass, copying negatives and zeros exactly.

    Returns a new float64 array. Each nonempty side's original total is kept,
    so zero-sum input stays zero-sum; zero sum is not required by this helper.
    Negative values, including values below -1, are the caller's responsibility.
    Single-sample and equal-magnitude sides are returned unchanged.
    """
    gamma_prime, epsilon = _parameters(gamma_prime, epsilon)
    result = _vector(shifts, "shifts").copy()
    _transform_positive(result, gamma_prime, epsilon)
    return result
