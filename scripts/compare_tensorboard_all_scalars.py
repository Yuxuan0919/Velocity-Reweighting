#!/usr/bin/env python3
"""Compare every TensorBoard scalar event bit-for-bit across two runs."""

import argparse
import hashlib
import math
from pathlib import Path
import struct

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


Scalar = tuple[int, bytes, float]


def load_run(run_dir: Path) -> dict[str, list[Scalar]]:
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found under {run_dir}")

    result: dict[str, list[Scalar]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = result.setdefault(tag, [])
            for event in accumulator.Scalars(tag):
                values.append((event.step, struct.pack(">f", event.value), event.value))
    if not result:
        raise ValueError(f"No TensorBoard scalar events found under {run_dir}")
    return result


def series_digest(values: list[Scalar]) -> str:
    payload = b"".join(struct.pack(">q", step) + bits for step, bits, _ in values)
    return hashlib.sha256(payload).hexdigest()


def describe_difference(tag: str, values_a: list[Scalar], values_b: list[Scalar]) -> str:
    common = min(len(values_a), len(values_b))
    for index in range(common):
        step_a, bits_a, value_a = values_a[index]
        step_b, bits_b, value_b = values_b[index]
        if step_a != step_b or bits_a != bits_b:
            return (
                f"tag={tag} index={index} "
                f"A(step={step_a},value={value_a},bits={bits_a.hex()}) "
                f"B(step={step_b},value={value_b},bits={bits_b.hex()})"
            )
    return f"tag={tag} length A={len(values_a)} B={len(values_b)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--expected-reward-steps", type=int)
    parser.add_argument("--expected-tag-count", type=int)
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    tags_a = set(run_a)
    tags_b = set(run_b)
    failed = False

    if tags_a != tags_b:
        print(f"FAIL tag sets: only_A={sorted(tags_a - tags_b)} only_B={sorted(tags_b - tags_a)}")
        failed = True
    if args.expected_tag_count is not None and len(tags_a) != args.expected_tag_count:
        print(f"FAIL expected {args.expected_tag_count} scalar tags; found {len(tags_a)}")
        failed = True

    differing_tags = []
    nonfinite = []
    for tag in sorted(tags_a | tags_b):
        values_a = run_a.get(tag, [])
        values_b = run_b.get(tag, [])
        for side, values in (("A", values_a), ("B", values_b)):
            for step, _, value in values:
                if not math.isfinite(value):
                    nonfinite.append((side, tag, step, value))
        comparable_a = [(step, bits) for step, bits, _ in values_a]
        comparable_b = [(step, bits) for step, bits, _ in values_b]
        if comparable_a != comparable_b:
            differing_tags.append(tag)
            print(f"FAIL {describe_difference(tag, values_a, values_b)}")

    if nonfinite:
        print(f"FAIL non-finite scalar values: {nonfinite[:10]}")
        failed = True

    if args.expected_reward_steps is not None:
        expected_steps = list(range(args.expected_reward_steps))
        for reward_tag in ("reward/pickscore", "reward/avg"):
            steps_a = [step for step, _, _ in run_a.get(reward_tag, [])]
            steps_b = [step for step, _, _ in run_b.get(reward_tag, [])]
            if steps_a != expected_steps or steps_b != expected_steps:
                print(
                    f"FAIL {reward_tag} expected steps 0..{args.expected_reward_steps - 1}; "
                    f"A={steps_a} B={steps_b}"
                )
                failed = True

    if differing_tags:
        first_steps = []
        for tag in differing_tags:
            values_a = run_a.get(tag, [])
            values_b = run_b.get(tag, [])
            for index in range(min(len(values_a), len(values_b))):
                if values_a[index][:2] != values_b[index][:2]:
                    first_steps.append(values_a[index][0])
                    break
        print(
            f"RESULT=DIVERGED differing_tags={len(differing_tags)}/{len(tags_a | tags_b)} "
            f"first_step={min(first_steps) if first_steps else 'unknown'}"
        )
        return 1

    if failed:
        return 1

    canonical = hashlib.sha256()
    for tag in sorted(run_a):
        canonical.update(tag.encode("utf-8") + b"\0")
        canonical.update(bytes.fromhex(series_digest(run_a[tag])))
    total_values = sum(len(values) for values in run_a.values())
    print(
        f"RESULT=BITWISE_IDENTICAL tags={len(run_a)} values={total_values} "
        f"sha256={canonical.hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
