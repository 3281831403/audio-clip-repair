"""Red tests for the audio clipping repair exercise.

Every test calls at least one public function from ``repair.clipping``.  The
stubs raise ``NotImplementedError`` at runtime, so the suite fails without any
collection, import, or syntax errors until the module is implemented.
"""

import math

import pytest

from repair.clipping import (
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

BASE_CONFIG = {
    "threshold": 0.9,
    "min_len": 2,
    "flat_tol": 0.01,
    "method": "linear",
    "ceiling": 1.0,
    "ratio": 1.0,
    "remove_dc": False,
}


def test_defect_01_noise_spike_is_not_a_clip():
    """A single above-threshold transient is not a full-scale flat top."""
    spike = [0.0, 0.2, 0.95, 0.2, 0.0]
    runs = find_runs(spike, 0.9)

    assert runs == [(2, 2, 0.95)]
    assert is_clipped(runs[0], min_len=1, flat_tol=0.01) is False

    plateau_runs = find_runs([0.0, 1.0, 1.0, 1.0, 0.0], 0.9)
    assert is_clipped(plateau_runs[0], min_len=1, flat_tol=0.01) is True


def test_defect_02_flat_top_requires_min_length():
    """Even an exact full-scale run needs enough samples to be a flat top."""
    assert is_clipped((3, 3, 1.0), min_len=2, flat_tol=0.01) is False
    assert is_clipped((3, 4, 1.0), min_len=2, flat_tol=0.01) is True


def test_defect_03_cubic_recovers_long_span_shape():
    """Cubic reconstruction must not sag like a straight line over a long run."""
    samples = [
        0.0,
        0.0,
        0.6,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.6,
        0.0,
        0.0,
    ]
    span = (3, 8)

    linear = interpolate_span(samples, span, "linear")
    cubic = interpolate_span(samples, span, "cubic")

    assert linear[5] == pytest.approx(0.6, abs=1e-6)
    assert cubic[5] > linear[5] + 0.01
    assert cubic[5] > 0.62


def test_defect_04_reconstructed_span_has_continuous_seams():
    """Rebuilt endpoints must meet the untouched neighbours without a jump."""
    samples = [0.0, 0.1, 0.2, 1.0, 1.0, 1.0, 0.3, 0.2, 0.0]
    rebuilt = interpolate_span(samples, (3, 5), "linear")

    assert rebuilt[:3] == samples[:3]
    assert rebuilt[6:] == samples[6:]
    assert abs(rebuilt[3] - rebuilt[2]) <= 0.05
    assert abs(rebuilt[5] - rebuilt[6]) <= 0.05


def test_defect_05_soft_limiter_is_not_hard_clipping():
    """Over-ceiling samples must stay below the ceiling and remain distinct."""
    samples = [-1.0, -0.75, -0.6, 0.0, 0.6, 0.75, 1.0]
    limited = limit_peak(samples, 0.5)

    assert limited[3] == 0.0
    assert all(abs(value) <= 0.5 + 1e-9 for value in limited)
    assert abs(limited[0]) < 0.5 - 1e-6
    assert abs(limited[0]) > abs(limited[1]) > abs(limited[2])
    assert abs(limited[4]) < abs(limited[5]) < abs(limited[6])


def test_defect_06_dynamics_restore_uses_local_bands():
    """Quiet and loud spans must get different gains from their own context."""
    samples = [
        0.10,
        -0.10,
        0.10,
        -0.10,
        0.10,
        1.0,
        1.0,
        1.0,
        0.10,
        -0.10,
        0.10,
        -0.10,
        0.10,
        0.80,
        -0.80,
        0.80,
        -0.80,
        1.0,
        1.0,
        1.0,
        0.80,
        -0.80,
        0.80,
        -0.80,
    ]
    original = list(samples)
    spans = [(5, 7), (17, 19)]

    restored = restore_dynamics(samples, spans, 1.0)

    quiet_level = sum(abs(value) for value in restored[5:8]) / 3.0
    loud_level = sum(abs(value) for value in restored[17:20]) / 3.0

    assert quiet_level == pytest.approx(0.1, abs=1e-6)
    assert loud_level == pytest.approx(0.8, abs=1e-6)
    assert loud_level > 4.0 * quiet_level
    assert samples == original
    assert restored[:5] == samples[:5]
    assert restored[8:17] == samples[8:17]
    assert restored[20:] == samples[20:]


def test_defect_07_dc_offset_ignores_silence():
    """A long silent block must not dilute the offset estimate by half."""
    tone = [
        0.2 + 0.3 * math.sin(2.0 * math.pi * n / 16.0) for n in range(64)
    ]
    samples = [0.0] * 64 + tone

    assert detect_dc_offset(samples) == pytest.approx(0.2, abs=1e-6)
    assert detect_dc_offset([0.0] * 16) == 0.0


def test_defect_08_overlapping_spans_are_merged():
    """Overlapping spans must be repaired once, like their merged union."""
    samples = [0.1, -0.1, 0.1] + [1.0] * 7 + [0.1, -0.1, 0.1]
    overlapping = [(3, 7), (5, 9)]
    merged = [(3, 9)]

    issues = validate(samples, overlapping)
    assert any(issue.startswith("overlap") for issue in issues)

    overlap_result = restore_dynamics(samples, overlapping, 1.0)
    merged_result = restore_dynamics(samples, merged, 1.0)

    assert overlap_result == pytest.approx(merged_result, abs=1e-9)


def test_defect_09_batch_tracks_do_not_share_buffers():
    """A repaired track must not leak its reconstruction into another track."""
    clipped = [0.1, -0.1, 0.1, 1.0, 1.0, 1.0, 0.1, -0.1, 0.1]
    clean = [0.2, -0.2, 0.2, 0.25, -0.25, 0.25]
    tracks = [
        {"name": "clipped", "samples": clipped},
        {"name": "clean", "samples": clean},
    ]

    batch = repair_batch(tracks, BASE_CONFIG)
    solo = repair_batch([tracks[0]], BASE_CONFIG)

    assert batch["tracks"][0]["samples"] == pytest.approx(
        solo["tracks"][0]["samples"], abs=1e-9
    )
    assert batch["tracks"][1]["samples"] == pytest.approx(clean, abs=1e-9)


def test_defect_10_whole_track_clipping_raises():
    """A fully clipped track must be rejected, never replaced by silence."""
    samples = [1.0] * 8
    good = [0.1, -0.1, 0.1, 1.0, 1.0, 0.1]
    before_good = list(good)
    issues = validate(samples, [(0, 7)])

    assert any(issue.startswith("all_clipped") for issue in issues)
    with pytest.raises(ClipRepairError):
        repair_batch(
            [
                {"name": "good", "samples": good},
                {"name": "bad", "samples": samples},
            ],
            BASE_CONFIG,
        )
    assert good == before_good


def test_clean_track_is_unchanged():
    """With no clipped runs the batch repair is transparent within 1e-9."""
    samples = [-0.5, 0.25, 0.0, 0.6, -0.2]
    result = repair_batch([{"name": "clean", "samples": samples}], BASE_CONFIG)

    assert result["tracks"][0]["samples"] == pytest.approx(samples, abs=1e-9)


def test_ceiling_one_is_identity():
    """The soft limiter must not color valid full-scale material at ceiling 1."""
    samples = [-1.0, -0.5, 0.0, 0.25, 1.0]
    limited = limit_peak(samples, 1.0)

    assert limited == pytest.approx(samples, abs=1e-9)


def test_empty_input_returns_empty():
    """An empty signal contains no above-threshold runs."""
    assert find_runs([], 0.9) == []


def test_min_len_below_one_raises():
    """A zero or negative minimum run length is invalid."""
    with pytest.raises(ClipRepairError):
        is_clipped((0, 2, 1.0), min_len=0, flat_tol=0.01)


def test_batch_preserves_track_order():
    """A successful batch must return tracks in their original order."""
    tracks = [
        {
            "name": "first",
            "samples": [0.1, -0.1, 1.0, 1.0, 0.1, -0.1],
        },
        {
            "name": "second",
            "samples": [0.2, -0.2, 0.2],
        },
        {
            "name": "third",
            "samples": [-0.1, 0.1, 1.0, 1.0, 1.0, -0.1, 0.1],
        },
    ]

    result = repair_batch(tracks, BASE_CONFIG)

    assert [track["name"] for track in result["tracks"]] == [
        "first",
        "second",
        "third",
    ]
    assert result["count"] == 3
