"""Failure taxonomy: the closed label set for episode outcomes.

The taxonomy is defined in full now so CSV schemas, reports, and tests stay
stable, but ONLY the labels in AUTO_LABELS are currently assigned by the
evaluation runner. The richer labels require reading env info flags
(is_grasped, is_obj_placed, object/goal distances) during evaluation —
Phase B instrumentation.

TODO(Janga): B5+ — assign richer labels in eval/runner.py from env info:
  wrong_object_selected  (a non-target cube moved to the goal region)
  reach_failure          (tcp never came within eps of the target object)
  grasp_failure          (reached but is_grasped never true)
  object_dropped         (is_grasped true then false, away from goal)
  placement_failure      (grasped and moved, final distance > threshold)
"""

from enum import Enum


class FailureType(str, Enum):
    SUCCESS = "success"
    LANGUAGE_GROUNDING_FAILURE = "language_grounding_failure"
    WRONG_OBJECT_SELECTED = "wrong_object_selected"
    REACH_FAILURE = "reach_failure"
    GRASP_FAILURE = "grasp_failure"
    OBJECT_DROPPED = "object_dropped"
    PLACEMENT_FAILURE = "placement_failure"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    ADAPTER_ACTION_MISMATCH = "adapter_action_mismatch"
    EMBODIMENT_REACHABILITY_LIMIT = "embodiment_reachability_limit"
    UNSTABLE_POLICY = "unstable_policy"
    SIMULATOR_ERROR = "simulator_error"
    UNKNOWN_FAILURE = "unknown_failure"


FAILURE_LABELS = frozenset(ft.value for ft in FailureType)

# Labels the runner can currently derive without extra instrumentation.
AUTO_LABELS = frozenset(
    {FailureType.SUCCESS.value, FailureType.TIMEOUT.value,
     FailureType.UNKNOWN_FAILURE.value, FailureType.SIMULATOR_ERROR.value}
)


def validate_failure_type(label: str) -> str:
    if label not in FAILURE_LABELS:
        known = ", ".join(sorted(FAILURE_LABELS))
        raise ValueError(
            f"unknown failure_type {label!r}; must be one of: {known}. "
            "Add new labels to xembench.analysis.taxonomy.FailureType, "
            "never ad hoc in data."
        )
    return label


def classify_episode(success: bool, truncated: bool) -> str:
    """Conservative auto-labeling from what the runner actually knows."""
    if success:
        return FailureType.SUCCESS.value
    if truncated:
        return FailureType.TIMEOUT.value
    # terminated early without success (rare in these envs)
    return FailureType.UNKNOWN_FAILURE.value
