"""Acceptance tests for the monotone shape-preserving handoff."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.adaptive_handoff import (
    NOMINAL_ANCHOR_HZ,
    MIN_TRANSITION_OCTAVES,
    MAX_TRANSITION_OCTAVES,
    STABILITY_WINDOW_OCTAVES,
    _smootherstep,
    _smootherstep_derivative,
    _smootherstep_second_derivative,
    _evaluate_monotone_cubic_bridge,
    _monotone_bridge,
    adaptive_masked_handoff,
    validate_locked_smootherstep,
)


def _log_grid(
    start_hz: float = 20.0,
    end_hz: float = 12000.0,
    points_per_octave: int = 192,
):
    octaves = math.log2(end_hz / start_hz)
    count = int(round(octaves * points_per_octave)) + 1
    return np.sort(
        np.unique(
            np.concatenate(
                [
                    np.geomspace(start_hz, end_hz, count),
                    np.array([1000.0]),
                ]
            )
        )
    )


def _smooth_base(freq):
    x = np.log2(freq / 1000.0)
    return 86.0 - 1.0 * x + 0.05 * x**2


def _make_masked(freq, end_hz=1200.0, amplitude=-2.0):
    x = np.log2(freq / 1000.0)
    span = math.log2(end_hz / 1000.0)
    t = np.clip(x / span, 0.0, 1.0)
    return _smooth_base(freq) + amplitude * _smootherstep(t)


def test_locked_constants():
    assert math.isclose(NOMINAL_ANCHOR_HZ, 1000.0, abs_tol=1e-12)
    assert math.isclose(MIN_TRANSITION_OCTAVES, 1.0 / 3.0, abs_tol=1e-12)
    assert math.isclose(MAX_TRANSITION_OCTAVES, 0.8, abs_tol=1e-12)
    assert math.isclose(STABILITY_WINDOW_OCTAVES, 0.20, abs_tol=1e-12)


def test_smootherstep_math_retained():
    t = np.linspace(0.0, 1.0, 100001)
    w = _smootherstep(t)
    dw = _smootherstep_derivative(t)
    d2w = _smootherstep_second_derivative(t)
    assert np.all(np.isfinite(w))
    assert float(np.min(w)) >= -1e-12
    assert float(np.max(w)) <= 1.0 + 1e-12
    assert float(np.min(dw)) >= -1e-12
    assert math.isclose(float(d2w[0]), 0.0, abs_tol=1e-15)
    assert math.isclose(float(d2w[-1]), 0.0, abs_tol=1e-15)
    assert validate_locked_smootherstep() is True


def test_handoff_preserves_base_target_through_1khz():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(freq, target, masked)

    below = freq <= 1000.0
    assert np.array_equal(output[below], target[below])
    assert diagnostics["actual_handoff_hz"] == 1000.0


def test_handoff_uses_masked_target_after_E():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(freq, target, masked)

    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    E = diagnostics["transition_end_hz"]
    after = freq > E
    assert np.array_equal(output[after], masked[after])


def test_selected_E_respects_locked_width():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    _, diagnostics = adaptive_masked_handoff(freq, target, masked)

    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    width = diagnostics["transition_width_octaves"]
    assert width >= MIN_TRANSITION_OCTAVES - 1e-12
    assert width <= MAX_TRANSITION_OCTAVES + 1e-12


def test_selection_is_earliest_feasible_E():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    _, diagnostics = adaptive_masked_handoff(freq, target, masked)

    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    assert diagnostics["selection_rule"] == "earliest_feasible_E"
    assert diagnostics["candidate_scoring"] is False
    assert diagnostics["pareto_selection"] is False
    assert diagnostics["curvature_optimization"] is False
    assert diagnostics["bridge_type"] == "monotone_cubic_hermite"


def test_handoff_is_deterministic():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    a, da = adaptive_masked_handoff(freq, target, masked)
    b, db = adaptive_masked_handoff(freq, target, masked)

    assert np.array_equal(a, b)
    assert da["transition_end_hz"] == db["transition_end_hz"]


def test_vertical_translation_invariance():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    a, da = adaptive_masked_handoff(freq, target, masked)
    b, db = adaptive_masked_handoff(freq, target + 37.25, masked + 37.25)

    assert da["transition_end_hz"] == db["transition_end_hz"]
    assert np.allclose(b, a + 37.25, rtol=0.0, atol=1e-10)


def test_positive_scale_invariance():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    a, da = adaptive_masked_handoff(freq, target, masked)
    b, db = adaptive_masked_handoff(freq, target * 3.0, masked * 3.0)

    assert da["transition_end_hz"] == db["transition_end_hz"]
    assert np.allclose(b, a * 3.0, rtol=0.0, atol=1e-9)


def test_no_stable_handoff_is_safe():
    freq = _log_grid()
    target = np.zeros_like(freq)
    masked = np.ones_like(freq)

    output, diagnostics = adaptive_masked_handoff(freq, target, masked)

    assert diagnostics["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(output, target)


def test_no_modification_below_H():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, _ = adaptive_masked_handoff(freq, target, masked)

    assert np.array_equal(output[freq < 1000.0], target[freq < 1000.0])


def test_no_modification_above_E_except_masked_target():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, d = adaptive_masked_handoff(freq, target, masked)
    assert d["status"] == "ADAPTIVE_HANDOFF"
    E = d["transition_end_hz"]
    assert np.array_equal(output[freq > E], masked[freq > E])


def test_monotone_bridge_compatibility_helper():
    result = _monotone_bridge(80.0, 78.0, -1.0, -1.0, 0.5)
    assert result["bridge_type"] == "monotone_cubic_hermite"
    assert result["shape_preserving"] is True


def test_bridge_accepts_realistic_synthetic_monotone_shape():
    freq = np.geomspace(1000.0, 1700.0, 97)
    x = np.log2(freq / 1000.0)
    target = 80.0 + 1.0 * x
    masked = target + 1.5 * _smootherstep(
        x / math.log2(1700.0 / 1000.0)
    )

    diagnostics = _evaluate_monotone_cubic_bridge(
        freq, target, masked, 0, len(freq) - 1
    )

    assert diagnostics["passed"]
    assert diagnostics["monotonicity_pass"]
    assert diagnostics["extrema_pass"]
    assert diagnostics["overshoot_pass"]
    assert diagnostics["undershoot_pass"]


def test_bridge_rejects_slope_reversal():
    freq = np.geomspace(1000.0, 1600.0, 97)
    x = np.log2(freq / 1000.0)
    u = x / x[-1]
    target = np.zeros_like(freq)
    # Positive endpoint delta but negative endpoint slope: the source geometry
    # itself turns back toward the endpoint, so the bridge must reject it.
    masked = -1.5 * u**3 + 1.5 * u**2 + u

    d = _evaluate_monotone_cubic_bridge(
        freq, target, masked, 0, len(freq) - 1
    )

    assert not d["passed"]


@pytest.mark.parametrize("amplitude", [-0.5, -1.0, -1.5, -2.0, -3.0])
def test_multiple_mask_amplitudes(amplitude):
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq, amplitude=amplitude)

    output, diagnostics = adaptive_masked_handoff(freq, target, masked)

    assert np.all(np.isfinite(output))
    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    assert diagnostics["bridge_pass"]


def test_locked_architecture_smoke():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(freq, target, masked)

    assert output.shape == target.shape
    assert np.all(np.isfinite(output))
    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    assert diagnostics["actual_handoff_hz"] == 1000.0
    assert diagnostics["selection_rule"] == "earliest_feasible_E"
    assert diagnostics["bridge_type"] == "monotone_cubic_hermite"
    assert diagnostics["bridge_pass"]
