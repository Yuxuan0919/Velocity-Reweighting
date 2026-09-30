"""Target-correction scale schedules and checkpoint compatibility checks."""

from dataclasses import dataclass
import math
from numbers import Integral


def _positive_scale(value, name):
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and strictly positive") from error
    if isinstance(value, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return scale


def _nonnegative_step(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class TargetScaleSchedule:
    """Keep c constant or decrease it linearly over a fixed step interval.

    Steps use the trainer's global_step convention, starting at zero. The
    initial scale applies through start_step, and the final scale applies
    from end_step onward. Constant schedules serialize only their active
    parameters, so unused interval settings do not change their identity.
    """

    initial_scale: float
    mode: str = "constant"
    final_scale: float | None = None
    start_step: int = 400
    end_step: int = 600

    def __post_init__(self):
        if self.mode not in ("constant", "linear_decay"):
            raise ValueError("mode must be constant or linear_decay")
        initial = _positive_scale(self.initial_scale, "initial_scale")
        start = _nonnegative_step(self.start_step, "start_step")
        end = _nonnegative_step(self.end_step, "end_step")
        if start >= end:
            raise ValueError("start_step must be less than end_step")
        final = self.final_scale
        if final is not None:
            final = _positive_scale(final, "final_scale")
            if final > initial:
                raise ValueError("final_scale must not exceed initial_scale")
        elif self.mode == "linear_decay":
            raise ValueError("linear_decay requires final_scale")
        object.__setattr__(self, "initial_scale", initial)
        object.__setattr__(self, "final_scale", final)
        object.__setattr__(self, "start_step", start)
        object.__setattr__(self, "end_step", end)

    def value_at(self, step):
        step = _nonnegative_step(step, "step")
        if self.mode == "constant" or step <= self.start_step:
            return self.initial_scale
        if step >= self.end_step:
            return self.final_scale
        fraction = (step - self.start_step) / (self.end_step - self.start_step)
        return self.initial_scale + (self.final_scale - self.initial_scale) * fraction

    def metadata(self):
        result = {"mode": self.mode, "initial_scale": self.initial_scale}
        if self.mode == "linear_decay":
            result.update(
                final_scale=self.final_scale,
                start_step=self.start_step,
                end_step=self.end_step,
            )
        return result


def validate_resume_schedule(training_state, requested_schedule):
    """Reject a changed schedule; older checkpoints imply a constant scale.

    ``requested_schedule`` is a TargetScaleSchedule instance. Legacy state
    without target_scale_schedule uses linear_target_scale, defaulting to 1.
    This check validates the schedule, not the checkpoint's global_step.
    """
    if not isinstance(requested_schedule, TargetScaleSchedule):
        raise ValueError("requested_schedule must be a TargetScaleSchedule")
    if "target_scale_schedule" not in training_state:
        saved = TargetScaleSchedule(
            initial_scale=training_state.get("linear_target_scale", 1.0)
        )
    else:
        metadata = training_state["target_scale_schedule"]
        if not isinstance(metadata, dict) or not {"mode", "initial_scale"} <= metadata.keys():
            raise ValueError("Checkpoint target_scale_schedule metadata is invalid")
        try:
            saved = TargetScaleSchedule(**metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("Checkpoint target_scale_schedule metadata is invalid") from error
    if saved.metadata() != requested_schedule.metadata():
        raise ValueError(
            "Checkpoint target scale schedule does not match this run: "
            f"saved={saved.metadata()}; requested={requested_schedule.metadata()}"
        )
