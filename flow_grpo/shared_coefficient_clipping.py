"""Exponential AWR mass shifts with one clipping coefficient per prompt group."""

from collections import defaultdict

import numpy as np


def compute_shared_exp_shifts(
    rewards,
    rollout_batch_ids,
    prompt_indices,
    group_size,
    *,
    variant="upper",
    epsilon_plus=2.0,
    epsilon_minus=1.0,
):
    """Return b = c_eff * (K * softmax_group(r / sigma_R) - 1).

    ``sigma_R`` is the population standard deviation of the complete gathered
    rollout, floored at 1e-6. Groups are identified by both rollout batch and
    dataset prompt index, so repeated prompts in different batches stay separate.
    The experiment fixes c = alpha = 1 and applies no variance normalization.

    ``upper`` only constrains the positive side; ``two_sided`` also constrains
    the negative side. ``unclipped`` returns the raw mass shifts. Diagnostics
    compare upper-only and two-sided coefficients at the supplied thresholds
    for every variant, regardless of which coefficient is used for training.
    Counterfactual diagnostics also check the experiment's 0.6 and 0.3 negative
    thresholds at the same positive threshold. The returned float32 vector
    preserves the gathered sample order and should be stored unchanged for all
    inner epochs of this rollout.
    """
    if variant not in {"unclipped", "upper", "two_sided"}:
        raise ValueError("variant must be unclipped, upper, or two_sided")
    if (
        isinstance(group_size, (bool, np.bool_))
        or not isinstance(group_size, (int, np.integer))
        or group_size < 2
    ):
        raise ValueError("group_size must be an integer of at least 2")
    if not np.isfinite(epsilon_plus) or epsilon_plus <= 0:
        raise ValueError("epsilon_plus must be finite and positive")
    if not np.isfinite(epsilon_minus) or not 0 < epsilon_minus <= 1:
        raise ValueError("epsilon_minus must be finite and in (0, 1]")

    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    rollout_batch_ids = np.asarray(rollout_batch_ids).reshape(-1)
    prompt_indices = np.asarray(prompt_indices).reshape(-1)
    if not (len(rewards) == len(rollout_batch_ids) == len(prompt_indices)):
        raise ValueError("Rewards and group identifiers must have the same length")
    if not rewards.size or not np.all(np.isfinite(rewards)):
        raise ValueError("A nonempty rollout with finite rewards is required")

    groups = defaultdict(list)
    for sample_idx, key in enumerate(zip(rollout_batch_ids, prompt_indices)):
        groups[key].append(sample_idx)
    for key, indices in groups.items():
        if len(indices) != group_size:
            raise ValueError(f"Prompt group {key} has {len(indices)} samples; expected {group_size}")

    gamma = max(float(rewards.std()), 1e-6)
    if not np.isfinite(gamma):
        raise ValueError("Rollout reward standard deviation must be finite")
    raw_shifts = np.empty_like(rewards)
    shifts = np.empty_like(rewards)
    coefficients = []
    extra_shrinkages = []
    counterfactual_shrinkages = {0.6: [], 0.3: []}
    zero_std_groups = 0
    for indices in groups.values():
        group_rewards = rewards[indices]
        zero_std_groups += int(np.all(group_rewards == group_rewards[0]))
        # Center before dividing to avoid overflow from a large reward offset.
        weights = np.exp((group_rewards - group_rewards.max()) / gamma)
        delta = weights / weights.mean() - 1.0
        d_plus = max(float(delta.max()), 0.0)
        d_minus = max(float(-delta.min()), 0.0)
        positive_limit = epsilon_plus / d_plus if d_plus > 0 else np.inf
        negative_limit = epsilon_minus / d_minus if d_minus > 0 else np.inf
        upper_coefficient = min(1.0, positive_limit)
        two_sided_coefficient = min(upper_coefficient, negative_limit)
        coefficient = {
            "unclipped": 1.0,
            "upper": upper_coefficient,
            "two_sided": two_sided_coefficient,
        }[variant]
        raw_shifts[indices] = delta
        shifts[indices] = coefficient * delta
        coefficients.append(coefficient)
        extra_shrinkages.append(upper_coefficient - two_sided_coefficient)
        for threshold, shrinkages in counterfactual_shrinkages.items():
            limit = threshold / d_minus if d_minus > 0 else np.inf
            shrinkages.append(upper_coefficient - min(upper_coefficient, limit))

    shifts = shifts.astype(np.float32)
    coefficients = np.asarray(coefficients)
    extra_shrinkages = np.asarray(extra_shrinkages)
    tolerance = 1e-8
    stats = {
        "gamma": gamma,
        "epsilon_plus": float(epsilon_plus),
        "epsilon_minus": float(epsilon_minus),
        "group_count": len(groups),
        "c_eff_mean": float(coefficients.mean()),
        "c_eff_min": float(coefficients.min()),
        "clipped_group_fraction": float(np.mean(coefficients < 1.0 - tolerance)),
        "negative_bound_active_fraction": float(np.mean(extra_shrinkages > tolerance)),
        "negative_bound_extra_shrink_mean": float(extra_shrinkages.mean()),
        "raw_shift_std": float(raw_shifts.std()),
        "raw_shift_min": float(raw_shifts.min()),
        "raw_shift_max": float(raw_shifts.max()),
        "b_min": float(shifts.min()),
        "b_max": float(shifts.max()),
        "b_mean_square": float(np.mean(shifts.astype(np.float64) ** 2)),
        "zero_sum_max_abs": max(
            abs(float(shifts[indices].sum(dtype=np.float64))) for indices in groups.values()
        ),
        "zero_std_group_ratio": zero_std_groups / len(groups),
    }
    for threshold, shrinkages in counterfactual_shrinkages.items():
        shrinkages = np.asarray(shrinkages)
        stats[f"negative_bound_{threshold}/active_fraction"] = float(np.mean(shrinkages > tolerance))
        stats[f"negative_bound_{threshold}/extra_shrink_mean"] = float(shrinkages.mean())
    return shifts, stats
