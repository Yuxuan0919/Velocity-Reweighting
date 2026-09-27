"""Groupwise mass-shift transformations used by the Step 2 experiments."""

import numpy as np


def square_both_sides(shifts):
    """Redistribute each sign's mass quadratically, with negative shifts >= -1.

    ``shifts`` is one prompt group's finite, one-dimensional Delta w array.
    Returns a new float64 array without modifying the input. Each sign keeps
    its total mass; zero-sum input therefore remains zero-sum, but callers may
    also supply a group containing only one sign. Negative inputs below -1
    are invalid because they cannot represent nonnegative original weights.

    Negative magnitudes solve sum(min(1, tau * magnitude**2)) = original mass.
    Saturated entries are fixed before the remaining mass is redistributed.
    Squaring ratios to the largest active magnitude avoids unnecessary
    overflow/underflow. Values below float64's representable range may round
    to zero, as with any floating-point probability allocation.
    """
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 1:
        raise ValueError("shifts must be a one-dimensional prompt group")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("shifts must be finite")
    if np.any(shifts < -1.0):
        raise ValueError("negative shifts must be at least -1")

    result = np.zeros_like(shifts)
    positive = shifts > 0
    if np.any(positive):
        magnitudes = shifts[positive]
        with np.errstate(over="ignore"):
            mass = magnitudes.sum()
        if not np.isfinite(mass):
            raise ValueError("total positive mass must be finite")
        weights = np.square(magnitudes / magnitudes.max())
        result[positive] = mass * (weights / weights.sum())

    negative_indices = np.flatnonzero(shifts < 0)
    magnitudes = -shifts[negative_indices]
    remaining_mass = magnitudes.sum()
    active = np.arange(magnitudes.size)
    while active.size:
        if remaining_mass == active.size:
            result[negative_indices[active]] = -1.0
            break
        active_magnitudes = magnitudes[active]
        weights = np.square(active_magnitudes / active_magnitudes.max())
        proposed = remaining_mass * (weights / weights.sum())
        saturated = proposed > 1.0
        if not np.any(saturated):
            result[negative_indices[active]] = -proposed
            break
        result[negative_indices[active[saturated]]] = -1.0
        remaining_mass -= np.count_nonzero(saturated)
        active = active[~saturated]

    return result
