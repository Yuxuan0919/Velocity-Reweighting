#!/usr/bin/env python3
"""One discrete heatmap for all weight variants in variants.tex."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reward_to_w_discrete_heatmap.png"

# One synthetic prompt group, sorted in descending reward order.  The repeated
# 0.51 and 0.37 rewards exercise the equal-boundary rule for Top-6 and Top-12.
REWARDS = np.array(
    [
        0.87, 0.75, 0.67, 0.61, 0.51, 0.51, 0.51, 0.45,
        0.43, 0.41, 0.39, 0.37, 0.37, 0.37, 0.33, 0.31,
        0.27, 0.25, 0.23, 0.17, 0.13, -0.10, -0.30, -1.20,
    ],
    dtype=np.float64,
)


def normalize(raw_values: np.ndarray) -> np.ndarray:
    """variants.tex: w_i = f(R_i) / mean_group(f(R))."""
    mean = float(raw_values.mean())
    if mean <= 0 or not np.isfinite(mean):
        raise ValueError("The group mean of f(R) must be positive and finite")
    return raw_values / mean


def softmax_weights(rewards: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = rewards / temperature
    # Subtracting the maximum leaves normalized w unchanged and avoids overflow.
    return normalize(np.exp(logits - logits.max()))


def soft_top_k_weights(rewards: np.ndarray, k: int) -> np.ndarray:
    """Assign boundary ties equally, exactly as in Experiment C."""
    group_size = len(rewards)
    if not 1 <= k <= group_size:
        raise ValueError("k must be between 1 and G")
    threshold = np.sort(rewards)[::-1][k - 1]
    above = rewards > threshold
    equal = rewards == threshold
    weights = np.zeros(group_size, dtype=np.float64)
    weights[above] = group_size / k
    weights[equal] = (group_size / k) * (k - above.sum()) / equal.sum()
    return weights


def build_rows(rewards: np.ndarray) -> tuple[list[str], np.ndarray]:
    mean = float(rewards.mean())
    std = float(rewards.std())  # Population std, as defined in variants.tex.
    centered_l1 = float(np.abs(rewards - mean).sum())
    labels: list[str] = []
    rows: list[np.ndarray] = []

    for c in (0.5, 1.0, 1.5, 2.5):
        labels.append(rf"Softmax $c\sigma$   $c={c:g}$")
        rows.append(softmax_weights(rewards, c * std))
    for c in (0.5, 1.0, 1.5, 2.5):
        labels.append(rf"Softmax $c\|R-\bar R\|_1$   $c={c:g}$")
        rows.append(softmax_weights(rewards, c * centered_l1))

    labels.append(r"Linear $[R]_+$")
    rows.append(normalize(np.maximum(rewards, 0.0)))
    labels.append(r"Linear $[1+R]_+$")
    rows.append(normalize(np.maximum(1.0 + rewards, 0.0)))

    for k in (6, 12):
        labels.append(rf"Soft Top-$k$   $k={k}$")
        rows.append(soft_top_k_weights(rewards, k))

    values = np.vstack(rows)
    np.testing.assert_allclose(values.mean(axis=1), 1.0, atol=1e-12)
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    return labels, values


def annotation(value: float, raw_reward: bool = False) -> str:
    if raw_reward:
        return f"{value:.2f}"
    if value == 0.0:
        return "0"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}"


def text_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.47 else "#172233"


def render(rewards: np.ndarray, labels: list[str], weights: np.ndarray) -> None:
    # Discrete bins: blue means w<1, near-white means w≈1, red means w>1.
    edges = np.array(
        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.3,
         1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0],
        dtype=np.float64,
    )
    colors = [
        "#112b55", "#183e72", "#1e5590", "#2c71aa", "#d6edf2",
        "#f7f7f7", "#f8d5c8", "#f2aaa0", "#de6a62", "#d35e58",
        "#bb3b45", "#a52b3a", "#8b1e2d", "#6d1025",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(edges, cmap.N)
    reward_cmap = colormaps["RdBu_r"]
    reward_norm = TwoSlopeNorm(
        vmin=float(rewards.min()), vcenter=0.0, vmax=float(rewards.max())
    )

    row_count = len(labels) + 1
    fig, ax = plt.subplots(figsize=(22.5, 10.5))  # Exactly one heatmap axis.

    for row in range(row_count):
        for column in range(len(rewards)):
            raw_row = row == 0
            value = rewards[column] if raw_row else weights[row - 1, column]
            color = (
                reward_cmap(reward_norm(value))
                if raw_row else cmap(norm(value))
            )
            ax.add_patch(
                Rectangle(
                    (column, row), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=0.65,
                )
            )
            ax.text(
                column + 0.5, row + 0.5, annotation(value, raw_row),
                ha="center", va="center", fontsize=6.15,
                color=text_color(color),
            )

    for boundary in (1, 5, 9, 11):
        ax.axhline(boundary, color="#253241", linewidth=1.25, zorder=4)

    ax.set_xlim(0, len(rewards))
    ax.set_ylim(row_count, 0)
    ax.set_xticks(np.arange(len(rewards)) + 0.5)
    ax.set_xticklabels(np.arange(1, len(rewards) + 1), fontsize=8)
    ax.set_yticks(np.arange(row_count) + 0.5)
    ax.set_yticklabels([r"Raw reward $R$", *labels], fontsize=10)
    ax.set_xlabel("Sample rank (same reward-sorted sample in every row)", fontsize=11, labelpad=9)
    ax.tick_params(axis="both", length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        boundaries=edges, ticks=(0, 0.5, 1, 1.5, 2, 3, 4, 6),
        fraction=0.023, pad=0.012, spacing="proportional",
    )
    colorbar.set_label(r"Normalized $w$ (row mean = 1)", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        r"Reward $\rightarrow$ normalized $w$: softmax, linear, and Soft Top-$k$ variants",
        fontsize=16, fontweight="semibold", y=0.985,
    )
    fig.text(
        0.5, 0.945,
        f"Synthetic group: G=24, mean R={rewards.mean():.3f}, "
        f"{(rewards < 0).sum()} rewards below 0, "
        f"{(rewards < -1).sum()} below -1",
        ha="center", fontsize=10.5,
    )
    fig.text(
        0.5, 0.012,
        "Top-6 and Top-12 illustrate the equal-reward boundary rule "
        "(ties at R=0.51 and R=0.37). "
        "Blue = low, red = high. The raw R row uses its own 0-centered reward scale; "
        "all w rows share the discrete w scale.",
        ha="center", fontsize=9.2,
    )
    fig.subplots_adjust(left=0.205, right=0.94, top=0.88, bottom=0.105)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(REWARDS) != 24 or not np.all(REWARDS[:-1] >= REWARDS[1:]):
        raise ValueError("Expected 24 reward-sorted samples")
    labels, weights = build_rows(REWARDS)
    render(REWARDS, labels, weights)
    print(OUTPUT)


if __name__ == "__main__":
    main()
