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


def exponential_both_sides(shifts, gamma_prime=1.0, epsilon=1e-6):
    """Scheme 4: exponential mass redistribution with negative shifts >= -1.

    Each side estimates its scale separately. The negative scale and weights
    are computed once from the full original negative side and held fixed
    during saturation. The multiplier solved below is for centered weights:
    tau_centered = tau_original * exp(gamma_minus * max(original_magnitudes)).
    Returns a new float64 array without modifying the input. Zero-sum input
    stays zero-sum, but the helper also accepts groups with only one sign.
    """
    gamma_prime, epsilon = _parameters(gamma_prime, epsilon)
    result = _vector(shifts, "shifts").copy()
    if np.any(result < -1.0):
        raise ValueError("negative shifts must be at least -1")
    _transform_positive(result, gamma_prime, epsilon)

    indices = np.flatnonzero(result < 0)
    magnitudes = -result[indices]
    if magnitudes.size < 2 or np.all(magnitudes == magnitudes[0]):
        return result
    # These are the ORIGINAL side's weights; never recompute gamma on active.
    weights, _, _ = l1_exponential_weights(magnitudes, gamma_prime, epsilon)
    remaining_mass = magnitudes.sum()
    active = np.arange(magnitudes.size)
    while active.size:
        if remaining_mass == active.size:
            result[indices[active]] = -1.0
            break
        proposed = remaining_mass * (weights[active] / weights[active].sum())
        saturated = proposed > 1.0
        if not np.any(saturated):
            result[indices[active]] = -proposed
            break
        result[indices[active[saturated]]] = -1.0
        remaining_mass -= np.count_nonzero(saturated)
        active = active[~saturated]
    return result
