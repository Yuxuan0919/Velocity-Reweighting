"""Prompt-group reward mappings from assets/softmax_variant/variants.tex.

These functions consume raw reward-function outputs, not NFT advantages.
All returned weights have mean one within each (rollout batch, prompt) group.
"""

from collections import defaultdict

import numpy as np


def validate_weight_settings(variant, c, p):
    if variant not in ("A", "B", "C", "D", "E", "F"):
        raise ValueError(f"Unknown weight variant: {variant!r}")
    if not np.isfinite(c) or not np.isfinite(p):
        raise ValueError("c and p must be finite")
    if variant == "A" and not 0.0 <= c <= 1.0:
        raise ValueError("Experiment A requires 0 <= c <= 1")
    if variant in ("B", "D", "F") and c <= 0.0:
        raise ValueError(f"Experiment {variant} requires c > 0")
    if variant == "E" and p <= 0.0:
        raise ValueError("Experiment E requires p > 0")


def _normalize_positive(values):
    """Compute f / mean(f); the all-zero case has no preferred sample."""
    maximum = values.max()
    if maximum == 0.0:
        # C/E are undefined when every raw reward is nonpositive.
        # Use w=1, i.e. zero reward-driven target shift, as in A's fallback.
        return np.ones_like(values)
    relative = values / maximum
    return relative / relative.mean()


def _normalize_log_values(log_values):
    relative = np.exp(log_values - log_values.max())
    return relative / relative.mean()


def weights_near_max(rewards, c):
    """A: uniform mass on R >= (1-c) Rmax; w=1 if Rmax <= 0."""
    maximum = rewards.max()
    if maximum <= 0.0:
        return np.ones_like(rewards)
    selected = rewards >= (1.0 - c) * maximum
    return selected.astype(np.float64) * (len(rewards) / selected.sum())


def weights_softmax_std(rewards, c):
    """B: exp(R / (c * population_std(R))) / mean_group(exp(...))."""
    # Subtracting the maximum cancels in normalization and avoids exp overflow.
    # Rescale before computing std to avoid squaring very small/large rewards.
    shifted = rewards - rewards.max()
    spread = -shifted.min()
    if spread == 0.0:
        return np.ones_like(rewards)
    relative = shifted / spread
    scale = relative.std(ddof=0)
    return _normalize_log_values(relative / (c * scale))


def weights_linear(rewards):
    """C: [R]+ / mean_group([R]+); no shifted-linear experiment."""
    return _normalize_positive(np.maximum(rewards, 0.0))


def weights_sigmoid(rewards, c):
    """D: sigmoid((R-0.5)/c) / mean_group(sigmoid((R-0.5)/c))."""
    logits = (rewards - 0.5) / c
    # Log-sigmoid avoids all-zero underflow when the entire group is far
    # below 0.5 (especially c=0.01), without changing reward ratios.
    log_values = -np.logaddexp(0.0, -logits)
    return _normalize_log_values(log_values)


def weights_power(rewards, p):
    """E: [R]+**p / mean_group([R]+**p)."""
    positive = np.maximum(rewards, 0.0)
    maximum = positive.max()
    if maximum == 0.0:
        return np.ones_like(rewards)
    # Scaling by maximum before the power cancels in group normalization.
    return _normalize_positive((positive / maximum) ** p)


def weights_softmax_l1(rewards, c):
    """F: softmax using c * mean(abs(R - mean(R))), not an L1 sum."""
    shifted = rewards - rewards.max()
    spread = -shifted.min()
    if spread == 0.0:
        return np.ones_like(rewards)
    relative = shifted / spread
    scale = np.abs(relative - relative.mean()).mean()
    return _normalize_log_values(relative / (c * scale))


def compute_variant_weights(
    prompts, rollout_batch_ids, raw_rewards, variant, c=1.0, p=1.0,
    expected_group_size=None,
):
    """Return weights in the original rank-major gathered-sample order.

    The same prompt in two rollout batches forms two independent groups.
    raw_rewards may be [N] or the base trainer's repeated [N, T] array.
    """
    validate_weight_settings(variant, c, p)
    prompts = np.asarray(prompts)
    rollout_batch_ids = np.asarray(rollout_batch_ids)
    rewards = np.asarray(raw_rewards, dtype=np.float64)
    if rewards.ndim == 2 and rewards.shape[1] > 0:
        if not np.all(rewards == rewards[:, :1]):
            raise ValueError("Repeated timestep rewards must be identical per sample")
        rewards = rewards[:, 0]
    if prompts.ndim != 1 or rollout_batch_ids.ndim != 1 or rewards.ndim != 1:
        raise ValueError("Expected 1D prompts/batch IDs and [N] or repeated [N,T] rewards")
    if not (len(prompts) == len(rollout_batch_ids) == len(rewards)) or len(rewards) == 0:
        raise ValueError("prompts, rollout_batch_ids and rewards need equal nonzero lengths")
    if not np.isfinite(rewards).all():
        raise ValueError("Raw rewards must be finite")
    if expected_group_size is not None and expected_group_size < 1:
        raise ValueError("expected_group_size must be positive")

    groups = defaultdict(list)
    for index, (batch_id, prompt) in enumerate(zip(rollout_batch_ids, prompts, strict=True)):
        groups[(int(batch_id), str(prompt))].append(index)

    weights = np.empty_like(rewards)
    for key, indices in groups.items():
        if expected_group_size is not None and len(indices) != expected_group_size:
            raise ValueError(
                f"Group {key!r} has {len(indices)} samples, expected {expected_group_size}; "
                "check distributed grouping and duplicate/truncated prompts"
            )
        group_rewards = rewards[indices]
        if variant == "A":
            group_weights = weights_near_max(group_rewards, c)
        elif variant == "B":
            group_weights = weights_softmax_std(group_rewards, c)
        elif variant == "C":
            group_weights = weights_linear(group_rewards)
        elif variant == "D":
            group_weights = weights_sigmoid(group_rewards, c)
        elif variant == "E":
            group_weights = weights_power(group_rewards, p)
        else:
            group_weights = weights_softmax_l1(group_rewards, c)
        if not np.isfinite(group_weights).all():
            raise ValueError(f"Nonfinite weights in experiment {variant}, group {key!r}")
        weights[indices] = group_weights
    return weights
