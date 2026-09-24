#!/usr/bin/env python3
"""Rewrite TensorBoard event files so that they contain scalar curves only."""

from __future__ import annotations

import argparse
import os
import stat
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter, masked_crc32c


# Scalar events are normally below 1 KiB. Records above this limit cannot be
# useful scalar curves, and seeking over them avoids transferring multi-MiB
# encoded images from shared storage.
MAX_SCALAR_RECORD_BYTES = 4 * 1024 * 1024


def event_records(path: Path) -> Iterator[tuple[bytes | None, int]]:
    """Yield verified small TFRecord payloads and skip oversized payloads."""
    with path.open("rb") as source:
        while True:
            header = source.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"truncated TFRecord header in {path}")

            length_bytes = header[:8]
            expected_header_crc = struct.unpack("<I", header[8:])[0]
            if masked_crc32c(length_bytes) != expected_header_crc:
                raise ValueError(f"invalid TFRecord header checksum in {path}")
            payload_size = struct.unpack("<Q", length_bytes)[0]

            if payload_size > MAX_SCALAR_RECORD_BYTES:
                source.seek(payload_size, os.SEEK_CUR)
                footer = source.read(4)
                if len(footer) != 4:
                    raise ValueError(f"truncated TFRecord payload in {path}")
                yield None, payload_size + 16
                continue

            payload = source.read(payload_size)
            footer = source.read(4)
            if len(payload) != payload_size or len(footer) != 4:
                raise ValueError(f"truncated TFRecord payload in {path}")
            expected_payload_crc = struct.unpack("<I", footer)[0]
            if masked_crc32c(payload) != expected_payload_crc:
                raise ValueError(f"invalid TFRecord payload checksum in {path}")
            yield payload, payload_size + 16


def is_scalar(value) -> bool:
    """Return whether a Summary.Value is rendered as a scalar curve."""
    value_kind = value.WhichOneof("value")
    if value_kind == "simple_value":
        return True
    if value_kind == "tensor":
        return value.metadata.plugin_data.plugin_name == "scalars"
    return False


def value_type(value) -> str:
    value_kind = value.WhichOneof("value") or "unknown"
    plugin_name = value.metadata.plugin_data.plugin_name
    return f"{value_kind}:{plugin_name}" if plugin_name else value_kind


def filter_event_file(path: Path) -> dict[str, object]:
    source_stat = path.stat()
    temp_path = path.with_name(f".{path.name}.curves-only.{os.getpid()}.tmp")
    stats: dict[str, object] = {
        "input_bytes": source_stat.st_size,
        "records_read": 0,
        "records_written": 0,
        "values_kept": 0,
        "values_dropped": 0,
        "dropped_types": Counter(),
    }
    bytes_read = 0
    next_progress = 512 * 1024 * 1024
    started_at = time.monotonic()

    try:
        with temp_path.open("wb") as output:
            writer = RecordWriter(output)
            for raw_event, record_size in event_records(path):
                stats["records_read"] += 1
                bytes_read += record_size
                if raw_event is None:
                    stats["values_dropped"] += 1
                    stats["dropped_types"]["oversized_non_scalar_record"] += 1
                    continue
                event = event_pb2.Event.FromString(raw_event)

                if event.WhichOneof("what") != "summary":
                    writer.write(raw_event)
                    stats["records_written"] += 1
                else:
                    kept_values = []
                    for value in event.summary.value:
                        if is_scalar(value):
                            kept_values.append(value)
                            stats["values_kept"] += 1
                        else:
                            stats["values_dropped"] += 1
                            stats["dropped_types"][value_type(value)] += 1

                    if kept_values:
                        if len(kept_values) == len(event.summary.value):
                            writer.write(raw_event)
                        else:
                            del event.summary.value[:]
                            event.summary.value.extend(kept_values)
                            writer.write(event.SerializeToString())
                        stats["records_written"] += 1

                if bytes_read >= next_progress:
                    elapsed = max(time.monotonic() - started_at, 0.001)
                    print(
                        f"  {path}: read {bytes_read / (1024**3):.2f} GiB "
                        f"({bytes_read / elapsed / (1024**2):.1f} MiB/s)",
                        flush=True,
                    )
                    next_progress += 512 * 1024 * 1024

            writer.flush()
            os.fsync(output.fileno())

        current_stat = path.stat()
        if (
            current_stat.st_size != source_stat.st_size
            or current_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise RuntimeError(f"source changed while it was being filtered: {path}")

        os.chmod(temp_path, stat.S_IMODE(source_stat.st_mode))
        os.utime(
            temp_path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
        os.replace(temp_path, path)
        stats["output_bytes"] = path.stat().st_size
        return stats
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def event_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*")
        else:
            raise FileNotFoundError(path)
        files.extend(
            candidate
            for candidate in candidates
            if candidate.is_file()
            and not candidate.is_symlink()
            and "tfevents" in candidate.name
        )
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    files = event_files(args.paths)
    if not files:
        parser.error("no TensorBoard event files found")

    total_input = 0
    total_output = 0
    total_dropped = Counter()
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Filtering {path}", flush=True)
        result = filter_event_file(path)
        total_input += int(result["input_bytes"])
        total_output += int(result["output_bytes"])
        total_dropped.update(result["dropped_types"])
        print(
            f"  {result['input_bytes'] / (1024**2):.2f} MiB -> "
            f"{result['output_bytes'] / (1024**2):.2f} MiB; "
            f"kept {result['values_kept']} scalar values, "
            f"dropped {result['values_dropped']} non-scalar values",
            flush=True,
        )

    print(
        f"Done: {len(files)} files, {total_input / (1024**3):.3f} GiB -> "
        f"{total_output / (1024**3):.3f} GiB, dropped {dict(total_dropped)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
