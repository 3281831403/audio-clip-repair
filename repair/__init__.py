"""Offline audio clipping repair package."""

from .clipping import (
    ClipRepairError,
    detect_dc_offset,
    find_runs,
    interpolate_span,
    is_clipped,
    limit_peak,
    repair_batch,
    restore_dynamics,
    validate,
)

__all__ = [
    "ClipRepairError",
    "detect_dc_offset",
    "find_runs",
    "interpolate_span",
    "is_clipped",
    "limit_peak",
    "repair_batch",
    "restore_dynamics",
    "validate",
]
