"""Clipping detection and repair primitives for old recordings.

Conventions used by every public function in this module:

* A sample is a linear amplitude in ``[-1.0, 1.0]``: ``0.0`` is silence and
  ``1.0`` is full scale.  The one exception is :func:`limit_peak`, which also
  accepts already-over-range finite amplitudes because taming them is its job.
* A span is an inclusive ``(start, end)`` index pair unless stated otherwise.
  A run tuple returned by :func:`find_runs` is ``(start, end, peak)``; the
  functions that accept a span also accept such a triple and ignore ``peak``.
* Unless a stricter tolerance is stated, comparisons use an absolute
  tolerance of ``1e-6``.  No function mutates its input lists.
* Empty sample lists produce empty outputs where an empty output makes sense.
* Invalid parameters, malformed spans, non-finite samples, or out-of-range
  samples raise :class:`ClipRepairError`.
"""

import math
from collections.abc import Sequence

_TOL = 1e-6


class ClipRepairError(ValueError):
    """Raised for invalid clipping-repair parameters, spans, or samples."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_samples(samples, allow_over_range=False, allow_empty=True):
    """Validate and copy a sample sequence into a new list of floats."""
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ClipRepairError("samples must be a sequence of numbers")
    out = []
    for value in samples:
        if not _is_number(value) or not math.isfinite(value):
            raise ClipRepairError(f"sample is not a finite number: {value!r}")
        value = float(value)
        if not allow_over_range and abs(value) > 1.0 + _TOL:
            raise ClipRepairError(f"sample out of range [-1.0, 1.0]: {value!r}")
        out.append(value)
    if not allow_empty and not out:
        raise ClipRepairError("samples must not be empty")
    return out


def _parse_span(span):
    """Normalize a ``(start, end)`` pair or ``(start, end, peak)`` triple."""
    if (
        isinstance(span, (str, bytes))
        or not isinstance(span, Sequence)
        or len(span) not in (2, 3)
    ):
        raise ClipRepairError(f"malformed span: {span!r}")
    start, end = span[0], span[1]
    if not _is_number(start) or not _is_number(end):
        raise ClipRepairError(f"malformed span: {span!r}")
    if not float(start).is_integer() or not float(end).is_integer():
        raise ClipRepairError(f"span indices must be integers: {span!r}")
    start, end = int(start), int(end)
    if start < 0 or start > end:
        raise ClipRepairError(f"malformed span: {span!r}")
    return start, end


def _parse_run(run):
    """Normalize a ``(start, end, peak)`` run triple."""
    if (
        isinstance(run, (str, bytes))
        or not isinstance(run, Sequence)
        or len(run) != 3
    ):
        raise ClipRepairError(f"malformed run: {run!r}")
    start, end = _parse_span(run[:2])
    peak = run[2]
    if not _is_number(peak) or not math.isfinite(peak):
        raise ClipRepairError(f"malformed run peak: {run!r}")
    return start, end, float(peak)


def _merge_spans(spans):
    """Merge overlapping or adjacent ``(start, end)`` spans."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_runs(samples: list[float], threshold: float) -> list[tuple]:
    """Return maximal runs whose absolute amplitude exceeds ``threshold``."""
    if (
        not _is_number(threshold)
        or not math.isfinite(threshold)
        or threshold < -_TOL
        or threshold > 1.0 + _TOL
    ):
        raise ClipRepairError(f"invalid threshold: {threshold!r}")
    vals = _check_samples(samples)

    runs = []
    start = None
    peak = 0.0
    for index, value in enumerate(vals):
        if abs(value) > threshold + _TOL:
            if start is None:
                start = index
                peak = abs(value)
            else:
                peak = max(peak, abs(value))
        elif start is not None:
            runs.append((start, index - 1, peak))
            start = None
    if start is not None:
        runs.append((start, len(vals) - 1, peak))
    return runs


def is_clipped(run: tuple, min_len: int, flat_tol: float) -> bool:
    """Decide whether a candidate run is a clipped flat top."""
    if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 1:
        raise ClipRepairError(f"invalid min_len: {min_len!r}")
    if (
        not _is_number(flat_tol)
        or not math.isfinite(flat_tol)
        or flat_tol < -_TOL
        or flat_tol > 1.0 + _TOL
    ):
        raise ClipRepairError(f"invalid flat_tol: {flat_tol!r}")
    start, end, peak = _parse_run(run)

    long_enough = end - start + 1 >= min_len
    flat_at_full_scale = abs(peak) >= 1.0 - flat_tol - _TOL
    return long_enough and flat_at_full_scale


def interpolate_span(samples: list[float], span: tuple, method: str) -> list[float]:
    """Rebuild one clipped span while leaving all other samples untouched."""
    vals = _check_samples(samples, allow_empty=False)
    if method not in ("linear", "cubic"):
        raise ClipRepairError(f"unknown method: {method!r}")
    start, end = _parse_span(span)
    n = len(vals)
    if end >= n:
        raise ClipRepairError(f"span out of bounds: {span!r}")

    left = vals[start - 1] if start > 0 else vals[start]
    right = vals[end + 1] if end < n - 1 else vals[end]
    length = end - start + 1

    out = list(vals)
    if method == "linear":
        for i in range(start, end + 1):
            t = (i - start + 1) / (length + 1)
            out[i] = left + t * (right - left)
    else:
        p0 = vals[start - 2] if start >= 2 else left
        p1 = left
        p2 = right
        p3 = vals[end + 2] if end + 2 <= n - 1 else right
        m1 = (p2 - p0) / 2.0
        m2 = (p3 - p1) / 2.0
        for i in range(start, end + 1):
            t = (i - start + 1) / (length + 1)
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h02 = t3 - t2
            out[i] = h00 * p1 + h10 * m1 + h01 * p2 + h02 * m2
    return out


def _soft_limit(value, ceiling):
    knee = 0.75 * ceiling
    a = abs(value)
    if a <= knee:
        return value
    h = ceiling - knee
    return math.copysign(ceiling - h * math.exp(-(a - knee) / h), value)


def limit_peak(samples: list[float], ceiling: float) -> list[float]:
    """Soft-limit ``samples`` so that every output magnitude is at most the ceiling."""
    if (
        not _is_number(ceiling)
        or not math.isfinite(ceiling)
        or ceiling <= 0.0
        or ceiling > 1.0 + _TOL
    ):
        raise ClipRepairError(f"invalid ceiling: {ceiling!r}")
    vals = _check_samples(samples, allow_over_range=True)

    if math.isclose(ceiling, 1.0, abs_tol=_TOL):
        # Exact identity for valid full-scale material; only genuinely
        # over-range samples are soft-limited.
        return [v if abs(v) <= 1.0 else _soft_limit(v, 1.0) for v in vals]
    return [_soft_limit(v, ceiling) for v in vals]


def restore_dynamics(samples: list[float], spans: list[tuple], ratio: float) -> list[float]:
    """Restore level inside clipped spans from their local unclipped context."""
    if not _is_number(ratio) or not math.isfinite(ratio) or ratio <= 0.0:
        raise ClipRepairError(f"invalid ratio: {ratio!r}")
    vals = _check_samples(samples)
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        raise ClipRepairError("spans must be a sequence")
    n = len(vals)
    parsed = []
    for span in spans:
        start, end = _parse_span(span)
        if end >= n:
            raise ClipRepairError(f"span out of bounds: {span!r}")
        parsed.append((start, end))
    if not vals:
        return []
    merged = _merge_spans(parsed)

    def inside_any_span(index):
        return any(s <= index <= e for s, e in merged)

    # Compute every gain from the original samples before applying any.
    gains = []
    for start, end in merged:
        length = end - start + 1
        context = []
        i = start - 1
        while i >= 0 and start - i <= length and not inside_any_span(i):
            context.append(vals[i])
            i -= 1
        i = end + 1
        while i < n and i - end <= length and not inside_any_span(i):
            context.append(vals[i])
            i += 1
        if not context:
            raise ClipRepairError(
                f"span {(start, end)} has no local unclipped context"
            )
        local_rms = math.sqrt(sum(c * c for c in context) / len(context))
        span_rms = math.sqrt(
            sum(vals[i] * vals[i] for i in range(start, end + 1)) / length
        )
        if span_rms <= 1e-12:
            gains.append(None)  # effectively silent span: leave unchanged
        else:
            gains.append(ratio * local_rms / max(span_rms, 1e-12))

    out = list(vals)
    for (start, end), gain in zip(merged, gains):
        if gain is None:
            continue
        for i in range(start, end + 1):
            out[i] *= gain
    return out


def detect_dc_offset(samples: list[float]) -> float:
    """Return the mean DC offset of the non-silent part of ``samples``."""
    vals = _check_samples(samples, allow_over_range=True)
    kept = [v for v in vals if abs(v) > _TOL]
    if not kept:
        return 0.0
    return sum(kept) / len(kept)


def validate(samples: list[float], spans: list[tuple]) -> list[str]:
    """Return a list of problems with ``spans``, or ``[]`` when they are valid."""
    vals = _check_samples(samples, allow_over_range=True)
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        raise ClipRepairError("spans must be a sequence")
    n = len(vals)

    issues = []
    in_bounds = []
    for span in spans:
        try:
            start, end = _parse_span(span)
            if end >= n:
                raise ClipRepairError
            in_bounds.append((start, end))
        except ClipRepairError:
            issues.append(f"out_of_bounds: {span!r}")

    ordered = sorted(in_bounds)
    for current, following in zip(ordered, ordered[1:]):
        if following[0] <= current[1]:
            issues.append(f"overlap: {current} and {following}")

    merged = _merge_spans(ordered)
    if n > 0 and len(merged) == 1 and merged[0] == (0, n - 1):
        issues.append("all_clipped: spans cover every sample")

    return issues


def repair_batch(tracks: list[dict], config: dict) -> dict:
    """Repair every track independently and atomically."""
    if not isinstance(config, dict):
        raise ClipRepairError("config must be a dict")

    threshold = config.get("threshold", 0.9)
    min_len = config.get("min_len", 3)
    flat_tol = config.get("flat_tol", 0.01)
    method = config.get("method", "cubic")
    ceiling = config.get("ceiling", 0.99)
    ratio = config.get("ratio", 1.0)
    remove_dc = config.get("remove_dc", False)

    if (
        not _is_number(threshold)
        or not math.isfinite(threshold)
        or threshold < -_TOL
        or threshold > 1.0 + _TOL
    ):
        raise ClipRepairError(f"invalid threshold: {threshold!r}")
    if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 1:
        raise ClipRepairError(f"invalid min_len: {min_len!r}")
    if (
        not _is_number(flat_tol)
        or not math.isfinite(flat_tol)
        or flat_tol < -_TOL
        or flat_tol > 1.0 + _TOL
    ):
        raise ClipRepairError(f"invalid flat_tol: {flat_tol!r}")
    if method not in ("linear", "cubic"):
        raise ClipRepairError(f"unknown method: {method!r}")
    if (
        not _is_number(ceiling)
        or not math.isfinite(ceiling)
        or ceiling <= 0.0
        or ceiling > 1.0 + _TOL
    ):
        raise ClipRepairError(f"invalid ceiling: {ceiling!r}")
    if not _is_number(ratio) or not math.isfinite(ratio) or ratio <= 0.0:
        raise ClipRepairError(f"invalid ratio: {ratio!r}")
    if not isinstance(remove_dc, bool):
        raise ClipRepairError(f"invalid remove_dc: {remove_dc!r}")

    if isinstance(tracks, (str, bytes)) or not isinstance(tracks, Sequence):
        raise ClipRepairError("tracks must be a sequence")

    # Normalize track dictionaries.
    normalized = []
    for index, track in enumerate(tracks):
        if isinstance(track, dict):
            if "samples" not in track:
                raise ClipRepairError(f"track {index} has no samples")
            samples = track["samples"]
            name = track.get("name", f"track-{index}")
            spans = track.get("spans", None)
        elif isinstance(track, Sequence) and not isinstance(track, (str, bytes)):
            samples = track
            name = f"track-{index}"
            spans = None
        else:
            raise ClipRepairError(f"invalid track: {track!r}")
        if not isinstance(name, str):
            raise ClipRepairError(f"invalid track name: {name!r}")
        normalized.append((name, samples, spans))

    # Validation pass: every track is checked before any is processed, so a
    # bad track aborts the whole batch with all inputs untouched.
    prepared = []
    for name, samples, spans in normalized:
        vals = _check_samples(samples)
        if spans is None:
            runs = find_runs(vals, threshold)
            merged = _merge_spans(
                (s, e) for s, e, p in runs if is_clipped((s, e, p), min_len, flat_tol)
            )
        else:
            if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
                raise ClipRepairError("spans must be a sequence")
            parsed = []
            for span in spans:
                start, end = _parse_span(span)
                if end >= len(vals):
                    raise ClipRepairError(f"span out of bounds: {span!r}")
                parsed.append((start, end))
            merged = _merge_spans(parsed)
        if vals and len(merged) == 1 and merged[0] == (0, len(vals) - 1):
            raise ClipRepairError(f"track {name!r} is entirely clipped")
        prepared.append((name, vals, merged))

    # Processing pass: each track is repaired on its own private copy; no
    # reconstruction buffer or filter state is shared between tracks.
    out_tracks = []
    for name, vals, merged in prepared:
        work = list(vals)
        dc_offset = detect_dc_offset(work)
        if remove_dc:
            work = [v - dc_offset for v in work]
        for span in merged:
            work = interpolate_span(work, span, method)
        if merged:
            work = restore_dynamics(work, merged, ratio)
        work = limit_peak(work, ceiling)
        out_tracks.append(
            {
                "name": name,
                "samples": work,
                "spans": list(merged),
                "dc_offset": dc_offset,
            }
        )

    return {"tracks": out_tracks, "count": len(out_tracks)}
