"""Acceptance tests for the locked quintic Adaptive Masked-EarPrint handoff."""

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
    _evaluate_quintic_bridge,
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
    """Negative-slope base so the synthetic handoff has one monotone direction."""
    x = np.log2(freq / 1000.0)
    return 86.0 - 1.0 * x + 0.05 * x**2


def _make_masked(freq, end_hz=1700.0, amplitude=-2.0):
    """Construct a mask whose transition coordinate matches engine log-frequency t."""
    x = np.log2(freq / 1000.0)
    span = math.log2(end_hz / 1000.0)
    t = np.clip(x / span, 0.0, 1.0)
    return _smooth_base(freq) + amplitude * _smootherstep(t)


def test_smootherstep_exact_endpoints():
    t = np.array([0.0, 1.0])
    w = _smootherstep(t)
    dw = _smootherstep_derivative(t)
    d2w = _smootherstep_second_derivative(t)

    assert math.isclose(float(w[0]), 0.0, abs_tol=1e-15)
    assert math.isclose(float(w[1]), 1.0, abs_tol=1e-15)
    assert math.isclose(float(dw[0]), 0.0, abs_tol=1e-15)
    assert math.isclose(float(dw[1]), 0.0, abs_tol=1e-15)
    assert math.isclose(float(d2w[0]), 0.0, abs_tol=1e-15)
    assert math.isclose(float(d2w[1]), 0.0, abs_tol=1e-15)


def test_smootherstep_bounded_and_monotonic():
    t = np.linspace(0.0, 1.0, 100001)
    w = _smootherstep(t)
    dw = _smootherstep_derivative(t)

    assert np.all(np.isfinite(w))
    assert float(np.min(w)) >= -1e-12
    assert float(np.max(w)) <= 1.0 + 1e-12
    assert float(np.min(dw)) >= -1e-12


def test_locked_smootherstep_validator():
    assert validate_locked_smootherstep() is True


def test_locked_constants():
    assert math.isclose(NOMINAL_ANCHOR_HZ, 1000.0, abs_tol=1e-12)
    assert math.isclose(MIN_TRANSITION_OCTAVES, 1.0 / 3.0, abs_tol=1e-12)
    assert math.isclose(MAX_TRANSITION_OCTAVES, 0.8, abs_tol=1e-12)
    assert math.isclose(STABILITY_WINDOW_OCTAVES, 0.20, abs_tol=1e-12)


def test_quintic_bridge_basic_shape():
    freq = np.geomspace(1000.0, 1700.0, 97)
    x = np.log2(freq / 1000.0)
    target = 80.0 - 1.0 * x
    t = x / math.log2(1700.0 / 1000.0)
    masked = target - 2.0 * _smootherstep(t)

    diagnostics = _evaluate_quintic_bridge(
        freq, target, masked, 0, len(freq) - 1
    )

    assert diagnostics["passed"]
    assert diagnostics["finite_pass"]
    assert diagnostics["monotonicity_pass"]
    assert diagnostics["extrema_pass"]
    assert diagnostics["overshoot_pass"]
    assert diagnostics["undershoot_pass"]
    assert diagnostics["endpoint_weight_pass"]
    assert diagnostics["endpoint_derivative_pass"]
    assert not diagnostics["slope_reversal"]


def test_quintic_bridge_no_endpoint_overshoot():
    freq = np.geomspace(1000.0, 1800.0, 101)
    x = np.log2(freq / 1000.0)
    target = 80.0 - 1.0 * x
    t = x / math.log2(1800.0 / 1000.0)
    masked = target - 3.0 * _smootherstep(t)

    diagnostics = _evaluate_quintic_bridge(
        freq, target, masked, 0, len(freq) - 1
    )

    assert diagnostics["overshoot_db"] <= 1e-8
    assert diagnostics["undershoot_db"] <= 1e-8
    assert diagnostics["passed"]


def test_quintic_bridge_rejects_slope_reversal():
    freq = np.geomspace(1000.0, 1600.0, 97)
    x = np.log2(freq / 1000.0)
    u = x / x[-1]

    target = np.zeros_like(freq)
    # Endpoint is positive, but the curve rises then falls: a genuine
    # local reversal that the hard gate must reject.
    masked = 3.0 * u - 4.0 * u**2

    diagnostics = _evaluate_quintic_bridge(
        freq, target, masked, 0, len(freq) - 1
    )

    assert not diagnostics["passed"]
    assert diagnostics["slope_reversal"] or not diagnostics["monotonicity_pass"]


def test_handoff_preserves_base_target_through_1khz():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    below = freq <= 1000.0
    assert np.array_equal(output[below], target[below])
    assert diagnostics["actual_handoff_hz"] == 1000.0


def test_handoff_uses_masked_target_after_E():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"

    E = diagnostics["transition_end_hz"]
    after = freq > E
    assert np.array_equal(output[after], masked[after])


def test_selected_E_respects_locked_width():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    _, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    width = diagnostics["transition_width_octaves"]
    assert width >= MIN_TRANSITION_OCTAVES - 1e-12
    assert width <= MAX_TRANSITION_OCTAVES + 1e-12


def test_selection_is_earliest_feasible_E():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    _, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    assert diagnostics["selection_rule"] == "earliest_feasible_E"
    assert diagnostics["candidate_scoring"] is False
    assert diagnostics["pareto_selection"] is False
    assert diagnostics["curvature_optimization"] is False


def test_handoff_is_deterministic():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output_a, diag_a = adaptive_masked_handoff(freq, target, masked)
    output_b, diag_b = adaptive_masked_handoff(freq, target, masked)

    assert np.array_equal(output_a, output_b)
    assert diag_a["status"] == diag_b["status"]
    assert diag_a["transition_end_hz"] == diag_b["transition_end_hz"]


def test_vertical_translation_invariance():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output_a, diag_a = adaptive_masked_handoff(freq, target, masked)

    offset = 37.25
    output_b, diag_b = adaptive_masked_handoff(
        freq, target + offset, masked + offset
    )

    assert diag_a["actual_handoff_hz"] == diag_b["actual_handoff_hz"]
    assert diag_a["transition_end_hz"] == diag_b["transition_end_hz"]
    assert np.allclose(
        output_b, output_a + offset, rtol=0.0, atol=1e-10
    )


def test_positive_scale_invariance():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output_a, diag_a = adaptive_masked_handoff(freq, target, masked)

    scale = 3.0
    output_b, diag_b = adaptive_masked_handoff(
        freq, target * scale, masked * scale
    )

    assert diag_a["actual_handoff_hz"] == diag_b["actual_handoff_hz"]
    assert diag_a["transition_end_hz"] == diag_b["transition_end_hz"]
    assert np.allclose(
        output_b, output_a * scale, rtol=0.0, atol=1e-9
    )


def test_no_stable_handoff_is_safe():
    freq = _log_grid()
    target = np.zeros_like(freq)
    masked = np.ones_like(freq)

    output, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert diagnostics["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(output, target)


def test_no_modification_below_H():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, _ = adaptive_masked_handoff(freq, target, masked)

    assert np.array_equal(
        output[freq < 1000.0],
        target[freq < 1000.0],
    )


def test_H_endpoint_equals_base_target():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    h_idx = np.where(freq == 1000.0)[0][0]
    assert math.isclose(
        output[h_idx], target[h_idx], rel_tol=0.0, abs_tol=0.0
    )

    if diagnostics["status"] == "ADAPTIVE_HANDOFF":
        assert diagnostics["smootherstep_endpoint_weight_pass"]


def test_no_legacy_candidate_optimisation():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    _, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert diagnostics.get("candidate_scoring", False) is False
    assert diagnostics.get("pareto_selection", False) is False
    assert diagnostics.get("curvature_optimization", False) is False


def test_monotone_bridge_compatibility_helper():
    result = _monotone_bridge(
        80.0, 78.0, -1.0, -1.0, 0.5
    )

    assert result["bridge_type"] == "quintic_smootherstep"
    assert result["endpoint_weight_start"] == 0.0
    assert result["endpoint_weight_end"] == 1.0
    assert result["endpoint_weight_derivative_start"] == 0.0
    assert result["endpoint_weight_derivative_end"] == 0.0


@pytest.mark.parametrize(
    "amplitude",
    [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0],
)
def test_multiple_mask_amplitudes(amplitude):
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq, amplitude=amplitude)

    output, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert np.all(np.isfinite(output))

    if diagnostics["status"] == "ADAPTIVE_HANDOFF":
        assert diagnostics["actual_handoff_hz"] == 1000.0
        assert diagnostics["bridge_pass"]
        assert diagnostics["monotonicity_pass"]
        assert diagnostics["extrema_pass"]
        assert diagnostics["overshoot_pass"]
        assert diagnostics["undershoot_pass"]
        assert diagnostics["slope_reversal_pass"]


def test_smootherstep_dense_monotonicity():
    t = np.linspace(0.0, 1.0, 1_000_001)
    w = _smootherstep(t)
    diff = np.diff(w)

    assert float(np.min(diff)) >= -1e-14
    assert float(np.max(w)) <= 1.0 + 1e-14
    assert float(np.min(w)) >= -1e-14


def test_locked_architecture_smoke():
    freq = _log_grid()
    target = _smooth_base(freq)
    masked = _make_masked(freq)

    output, diagnostics = adaptive_masked_handoff(
        freq, target, masked
    )

    assert output.shape == target.shape
    assert output.shape == masked.shape
    assert np.all(np.isfinite(output))
    assert diagnostics["status"] == "ADAPTIVE_HANDOFF"
    assert diagnostics["actual_handoff_hz"] == 1000.0
    assert diagnostics["selection_rule"] == "earliest_feasible_E"
    assert diagnostics["bridge_type"] == "quintic_smootherstep"
    assert diagnostics["smootherstep_endpoint_weight_pass"]
    assert diagnostics["smootherstep_endpoint_derivative_pass"]
    assert diagnostics["bridge_pass"]
    assert diagnostics["candidate_scoring"] is False
    assert diagnostics["pareto_selection"] is False
    assert diagnostics["curvature_optimization"] is False