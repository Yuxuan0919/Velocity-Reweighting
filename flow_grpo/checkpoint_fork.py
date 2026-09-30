"""Validate the explicit Linear c=15, step-420 -> c=5 experiment fork.

Ordinary resume keeps its strict objective checks. This opt-in path changes
only the target scale while loading the complete original training state.
The callbacks keep checkpoint I/O in the trainer and allow CPU-only checks.
"""

from numbers import Integral
from pathlib import Path

from flow_grpo.target_scale import TargetScaleSchedule, validate_resume_schedule


def _overlap(first, second):
    return first == second or first in second.parents or second in first.parents


def _validate_step(state, *, source):
    step = state.get("global_step")
    epoch = state.get("epoch")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (step, epoch)):
        raise ValueError("Fork checkpoints require integer epoch and global_step")
    if epoch != step or (step != 420 if source else step < 420):
        raise ValueError(
            "Fork source must have epoch=global_step=420; continuation checkpoints "
            "must keep epoch=global_step>=420 (one optimizer attempt per round)"
        )


def resolve_linear_fork_checkpoint(
    source_checkpoint, save_dir, logdir, requested_resume, *, reward_mapping,
    mass_shift_transform, target_scale_schedule, find_checkpoint, is_complete, load_state,
):
    """Return (checkpoint path, state, provenance) without writing any files.

    A completed child checkpoint in the new output directory takes precedence
    over the original source. A rerun therefore continues its c=5 experiment
    even if the original checkpoint has since been moved off the machine.
    """
    if (
        reward_mapping != "linear"
        or mass_shift_transform != "none"
        or target_scale_schedule.metadata() != TargetScaleSchedule(5.0).metadata()
    ):
        raise ValueError("This fork requires Linear, no mass-shift transform, and constant c=5")
    source = Path(source_checkpoint).expanduser()
    if not source.is_absolute() or source.name != "checkpoint-420":
        raise ValueError("--fork_from_checkpoint must be an absolute path to checkpoint-420")
    source = source.resolve()
    # Check again after resolving a symlink so checkpoint-420 cannot hide a
    # different source checkpoint behind an alias.
    if source.name != "checkpoint-420":
        raise ValueError("The resolved fork source must be checkpoint-420")
    if not save_dir or not logdir:
        raise ValueError("Fork experiments require explicit, separate save_dir and logdir")
    source_run = source.parent.parent if source.parent.name == "checkpoints" else source.parent
    output = Path(save_dir).expanduser().resolve()
    logs = Path(logdir).expanduser().resolve()
    if _overlap(output, source_run) or _overlap(logs, source_run) or _overlap(output, logs):
        raise ValueError("Fork save/log directories must be separate from each other and the source run")

    provenance = {
        "experiment": "linear_c15_step420_to_c5",
        "source_checkpoint": str(source),
        "source_global_step": 420,
        "source_target_scale": 15.0,
        "target_scale": 5.0,
        "source_rng_restored": False,
    }
    candidate = find_checkpoint(requested_resume or str(output))
    if requested_resume and not candidate:
        raise ValueError(f"No complete fork continuation checkpoint at: {requested_resume}")
    if candidate:
        candidate = str(Path(candidate).resolve())
        if not (Path(candidate) / "_SUCCESS").is_file() or not is_complete(candidate):
            raise ValueError("Fork continuation requires a complete checkpoint with _SUCCESS")
        state = load_state(candidate)
        if state.get("checkpoint_fork") != provenance:
            raise ValueError("Continuation checkpoint does not belong to this step-420 fork")
        _validate_step(state, source=False)
        expected_scale = 5.0
    else:
        # A half-written child checkpoint should be investigated, not silently
        # replaced by a second fork from the original training state.
        if output.exists() and any(output.iterdir()):
            raise ValueError("Fork output is nonempty but has no complete checkpoint; refusing to restart from 420")
        if not (source / "_SUCCESS").is_file() or not is_complete(str(source)):
            raise ValueError(f"Fork source requires a complete checkpoint-420 with _SUCCESS: {source}")
        candidate = str(source)
        state = load_state(candidate)
        _validate_step(state, source=True)
        expected_scale = 15.0

    if (
        state.get("reward_mapping") != "linear"
        or state.get("linear_target_scale") != expected_scale
        or state.get("mass_shift_transform", "none") != "none"
    ):
        raise ValueError(f"Expected a Linear checkpoint with constant c={expected_scale:g} and no transform")
    validate_resume_schedule(state, TargetScaleSchedule(expected_scale))
    return candidate, state, provenance
