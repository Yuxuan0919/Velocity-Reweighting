"""Groupwise mass-shift transformations used by the Step 2 experiments."""

import numpy as np


def square_positive_shifts(shifts):
    """Square and renormalize positive shifts; leave nonpositive entries unchanged.

    Accepts one finite, one-dimensional prompt group and returns a new float64
    array. Positive mass is preserved without imposing an upper cap. Negative
    values and signed zeros are copied unchanged, including values below -1;
    validating original importance weights is the caller's responsibility.
    A zero-sum input remains zero-sum, but zero sum is not required here.

    Normalize by the largest positive magnitude before squaring to avoid
    unnecessary overflow/underflow. Allocations smaller than float64 can
    represent may still round to zero.
    """
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 1:
        raise ValueError("shifts must be a one-dimensional prompt group")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("shifts must be finite")

    result = shifts.copy()
    positive = shifts > 0
    if np.any(positive):
        magnitudes = shifts[positive]
        with np.errstate(over="ignore"):
            mass = magnitudes.sum()
        if not np.isfinite(mass):
            raise ValueError("total positive mass must be finite")
        weights = np.square(magnitudes / magnitudes.max())
        result[positive] = mass * (weights / weights.sum())
    return result
