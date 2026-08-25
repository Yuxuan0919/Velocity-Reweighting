#!/usr/bin/env python3
"""Visualize reward-to-weight transformations for several gamma values."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

# Keep Matplotlib's runtime cache in a writable location on shared servers.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diffusionnft-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GAMMAS = (0.01, 0.1, 1.0, 5.0)
SHARED_GAMMA_FILENAME = "shared_gamma_reward_transform_values.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot r, W=exp(r/gamma), omega=W/mean(W), and delta=omega-1."
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/reward_weight_experiment"),
    )
    return parser.parse_args()


def compute_values(rewards: np.ndarray, gamma: float) -> tuple[np.ndarray, ...]:
    """Apply the transformations exactly as specified by the experiment."""
    weights = np.exp(rewards / gamma)
    omega = weights / weights.mean()
    delta = omega - 1.0
    return weights, omega, delta


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}"


def plot_gamma(
    rewards: np.ndarray,
    gamma: float,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights, omega, delta = compute_values(rewards, gamma)
    sample_ids = np.arange(1, rewards.size + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    plots = (
        (rewards, r"Reward: $r \sim \mathcal{U}(0,1)$", "r", "#4C78A8"),
        (weights, rf"Weight: $W = \exp(r/\gamma)$, $\gamma={gamma_label(gamma)}$", "W", "#F58518"),
        (omega, r"Group-normalized weight: $\omega = W / \mathrm{mean}(W)$", r"$\omega$", "#54A24B"),
        (delta, r"Shifted value: $\delta = \omega - 1$", r"$\delta$", "#E45756"),
    )

    for ax, (values, title, ylabel, color) in zip(axes.flat, plots):
        ax.bar(sample_ids, values, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Sample index (sorted by r)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sample_ids)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylim(0.0, 1.0)
    axes[1, 0].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)

    fig.suptitle(
        rf"Reward transformation distributions ($n={rewards.size}$, $\gamma={gamma_label(gamma)}$)",
        fontsize=18,
        fontweight="bold",
    )
    output_path = output_dir / f"reward_transform_gamma_{gamma_label(gamma)}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return weights, omega, delta


def write_csv(
    rewards: np.ndarray,
    results: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    output_path = output_dir / "reward_transform_values.csv"
    fieldnames = ["sample_index", "r"]
    for gamma in results:
        suffix = gamma_label(gamma)
        fieldnames.extend((f"W_gamma_{suffix}", f"omega_gamma_{suffix}", f"delta_gamma_{suffix}"))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, reward in enumerate(rewards):
            row: dict[str, float | int] = {"sample_index": index + 1, "r": reward}
            for gamma, (weights, omega, delta) in results.items():
                suffix = gamma_label(gamma)
                row[f"W_gamma_{suffix}"] = weights[index]
                row[f"omega_gamma_{suffix}"] = omega[index]
                row[f"delta_gamma_{suffix}"] = delta[index]
            writer.writerow(row)


def sample_reward_groups(seed: int, num_samples: int) -> dict[str, np.ndarray]:
    """Sample three reward shapes used by the shared-global-std experiment."""
    rng = np.random.default_rng(seed)
    edge_left_count = num_samples // 2
    edge_right_count = num_samples - edge_left_count

    # A balanced mixture makes the two modes near 0 and 1 visible even for n=20.
    edge_rewards = np.concatenate(
        (
            rng.beta(1.0, 6.0, edge_left_count),
            1.0 - rng.beta(1.0, 6.0, edge_right_count),
        )
    )
    return {
        "uniform": rng.uniform(0.0, 1.0, num_samples),
        "edge_concentrated": edge_rewards,
        "center_concentrated": rng.beta(8.0, 8.0, num_samples),
    }


def plot_shared_gamma_group(
    rewards: np.ndarray,
    group_name: str,
    reward_title: str,
    gamma: float,
    total_samples: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot one reward group using the standard deviation of all groups."""
    rewards = np.sort(rewards)
    weights, omega, delta = compute_values(rewards, gamma)
    sample_ids = np.arange(1, rewards.size + 1)
    gamma_text = f"{gamma:.6f}"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots = (
        (rewards, reward_title, "r", "#4C78A8"),
        (
            weights,
            rf"Weight: $W = \exp(r/\gamma)$, $\gamma={gamma_text}$",
            "W",
            "#F58518",
        ),
        (
            omega,
            r"Group-normalized weight: $\omega = W / \mathrm{mean}(W)$",
            r"$\omega$",
            "#54A24B",
        ),
        (delta, r"Shifted value: $\delta = \omega - 1$", r"$\delta$", "#E45756"),
    )

    for ax, (values, title, ylabel, color) in zip(axes.flat, plots):
        ax.bar(sample_ids, values, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Sample index (sorted by r)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sample_ids)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylim(0.0, 1.0)
    axes[1, 0].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)

    fig.suptitle(
        rf"Shared global-std transformation: $\gamma=\mathrm{{std}}(r_{{\rm all\ {total_samples}}})={gamma_text}$",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path = output_dir / f"shared_gamma_{group_name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return weights, omega, delta


def run_shared_gamma_experiment(
    seed: int,
    num_samples: int,
    output_dir: Path,
) -> float:
    groups = sample_reward_groups(seed, num_samples)
    all_rewards = np.concatenate(tuple(groups.values()))
    global_gamma = float(np.std(all_rewards, ddof=0))
    total_samples = all_rewards.size
    reward_titles = {
        "uniform": r"Reward: $r \sim \mathcal{U}(0,1)$",
        "edge_concentrated": r"Reward: two-sided mixture concentrated near 0 and 1",
        "center_concentrated": r"Reward: $r \sim \mathrm{Beta}(8,8)$, concentrated near 0.5",
    }

    rows: list[dict[str, float | int | str]] = []
    for group_name, unsorted_rewards in groups.items():
        rewards = np.sort(unsorted_rewards)
        weights, omega, delta = plot_shared_gamma_group(
            rewards,
            group_name,
            reward_titles[group_name],
            global_gamma,
            total_samples,
            output_dir,
        )
        for index, (reward, weight, normalized, shifted) in enumerate(
            zip(rewards, weights, omega, delta), start=1
        ):
            rows.append(
                {
                    "distribution": group_name,
                    "sample_index": index,
                    "global_gamma": global_gamma,
                    "r": reward,
                    "W": weight,
                    "omega": normalized,
                    "delta": shifted,
                }
            )

    with (output_dir / SHARED_GAMMA_FILENAME).open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "distribution",
            "sample_index",
            "global_gamma",
            "r",
            "W",
            "omega",
            "delta",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return global_gamma


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if any(gamma <= 0 for gamma in args.gammas):
        raise ValueError("Every gamma must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rewards = np.sort(rng.uniform(0.0, 1.0, args.num_samples))

    results = {
        gamma: plot_gamma(rewards, gamma, args.output_dir) for gamma in args.gammas
    }
    write_csv(rewards, results, args.output_dir)

    global_gamma = run_shared_gamma_experiment(
        args.seed, args.num_samples, args.output_dir
    )

    print(f"Saved {len(results)} figures and one CSV to {args.output_dir.resolve()}")
    for gamma, (_, omega, delta) in results.items():
        print(
            f"gamma={gamma_label(gamma):>4}: "
            f"mean(omega)={omega.mean():.16f}, "
            f"mean(delta)={delta.mean():.16f}, "
            f"max(omega)={omega.max():.6f}"
        )
    print(
        "Saved 3 shared-global-std figures and one CSV: "
        f"gamma=std(all {3 * args.num_samples} rewards)={global_gamma:.6f}"
    )


if __name__ == "__main__":
    main()
