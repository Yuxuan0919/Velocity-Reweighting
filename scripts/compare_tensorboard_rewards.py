#!/usr/bin/env python3
"""Compare TensorBoard reward scalars bit-for-bit across two fresh runs."""

import argparse
import hashlib
from pathlib import Path
import struct

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = ("reward/pickscore", "reward/avg")


def load_scalars(run_dir: Path, tag: str) -> list[tuple[int, bytes, float]]:
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found under {run_dir}")

    values = []
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            value_bits = struct.pack(">f", event.value)
            values.append((event.step, value_bits, event.value))

    values.sort(key=lambda item: item[0])
    return values


def series_digest(values: list[tuple[int, bytes, float]]) -> str:
    payload = b"".join(struct.pack(">q", step) + bits for step, bits, _ in values)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--tag", action="append", dest="tags")
    args = parser.parse_args()

    expected_step_ids = list(range(args.expected_steps))
    failed = False
    for tag in args.tags or DEFAULT_TAGS:
        series_a = load_scalars(args.run_a, tag)
        series_b = load_scalars(args.run_b, tag)
        steps_a = [item[0] for item in series_a]
        steps_b = [item[0] for item in series_b]

        if steps_a != expected_step_ids or steps_b != expected_step_ids:
            print(
                f"FAIL {tag}: expected steps 0..{args.expected_steps - 1}; "
                f"run_a={steps_a[:3]}...{steps_a[-3:] if steps_a else []} ({len(steps_a)}), "
                f"run_b={steps_b[:3]}...{steps_b[-3:] if steps_b else []} ({len(steps_b)})"
            )
            failed = True
            continue

        bits_a = [item[1] for item in series_a]
        bits_b = [item[1] for item in series_b]
        if bits_a != bits_b:
            first_difference = next(index for index, pair in enumerate(zip(bits_a, bits_b)) if pair[0] != pair[1])
            value_a = series_a[first_difference][2]
            value_b = series_b[first_difference][2]
            print(
                f"FAIL {tag}: first difference at step {first_difference}: "
                f"{value_a} ({bits_a[first_difference].hex()}) != "
                f"{value_b} ({bits_b[first_difference].hex()})"
            )
            failed = True
            continue

        print(
            f"PASS {tag}: {len(series_a)} float32 values are bitwise identical; "
            f"sha256={series_digest(series_a)}; first={series_a[0][2]}; last={series_a[-1][2]}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
