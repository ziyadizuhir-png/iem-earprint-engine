import numpy as np
import pytest

from engine.adaptive_handoff import (
    adaptive_masked_handoff,
    _evaluate_exact_c1_bridge,
)


def _grid(end=5000.0):
    return np.geomspace(500.0, end, 80)


def _synthetic():
    f = _grid()
    x = np.log2(f / 1000.0)
    base = 0.8 * x
    # Smooth destination with a compatible positive slope.
    masked = 1.8 * x + 0.15 * np.tanh(2.0 * x)
    return f, base, masked


def test_locked_bounds_are_enforced():
    f, base, masked = _synthetic()

    with pytest.raises(ValueError):
        adaptive_masked_handoff(
            f, base, masked, min_transition_octaves=0.32
        )

    with pytest.raises(ValueError):
        adaptive_masked_handoff(
            f, base, masked, max_transition_octaves=0.81
        )


def test_exact_c1_retains_raw_endpoint_slopes():
    values, diag = _evaluate_exact_c1_bridge(
        0.0, 1.0, 1.0, 1.0, 1.0
    )
    assert np.isclose(diag["used_start_slope"], 1.0)
    assert np.isclose(diag["used_end_slope"], 1.0)
    assert diag["bridge_pass"] is True
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_alpha_beta_hard_gate():
    with pytest.raises(ValueError):
        _evaluate_exact_c1_bridge(0.0, 1.0, 3.5, 3.5, 1.0)


def test_earliest_feasible_candidate_wins():
    # Dense grid contains candidates just above 1/3 octave.
    f = np.geomspace(1000.0, 2600.0, 120)
    x = np.log2(f / 1000.0)
    base = 0.5 * x
    masked = 0.7 * x + 0.25 * x * x

    out, diag = adaptive_masked_handoff(
        f, base, masked,
        min_transition_octaves=1.0/3.0,
        max_transition_octaves=0.8,
        stability_window_octaves=0.2,
    )
    assert diag["status"] == "HANDOFF_ACCEPTED"
    assert diag["actual_handoff_hz"] == pytest.approx(
        min(
            f[i] for i in range(1, len(f))
            if 1.0/3.0 - 1e-12 <= np.log2(f[i]/1000.0) <= 0.8 + 1e-12
        )
    )
    assert np.all(np.isfinite(out))


def test_lower_bound_candidate_is_not_accepted():
    f = np.geomspace(1000.0, 1200.0, 40)
    x = np.log2(f / 1000.0)
    base = 0.5 * x
    masked = 0.7 * x + 0.1 * x*x
    _, diag = adaptive_masked_handoff(
        f, base, masked,
        min_transition_octaves=1.0/3.0,
        max_transition_octaves=0.8,
    )
    # Entire grid ends below 1/3 octave.
    assert diag["status"] == "NO_STABLE_HANDOFF"


def test_upper_bound_is_hard():
    f = np.geomspace(1000.0, 1800.0, 60)
    x = np.log2(f / 1000.0)
    base = 0.4 * x
    masked = 0.6 * x + 0.1*x*x
    _, diag = adaptive_masked_handoff(
        f, base, masked,
        min_transition_octaves=1.0/3.0,
        max_transition_octaves=0.8,
    )
    assert diag["status"] in {"HANDOFF_ACCEPTED", "NO_STABLE_HANDOFF"}
    if diag["status"] == "HANDOFF_ACCEPTED":
        assert np.log2(diag["actual_handoff_hz"] / 1000.0) <= 0.8 + 1e-12


def test_no_stable_handoff_fails_safe_to_base():
    f = _grid()
    base = np.zeros_like(f)
    masked = np.ones_like(f) * 3.0
    # Force impossible endpoint direction by making masked locally descend
    # from the anchor region.
    x = np.log2(f / 1000.0)
    masked = 2.0 - 2.0*x
    out, diag = adaptive_masked_handoff(f, base, masked)
    assert diag["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(out, base)


def test_diagnostics_schema_consistency():
    f, base, masked = _synthetic()
    _, diag = adaptive_masked_handoff(f, base, masked)
    required = {
        "status",
        "selection_rule",
    }
    assert required.issubset(diag)
    if diag["status"] == "HANDOFF_ACCEPTED":
        for key in (
            "destination_slope_stable",
            "destination_curvature_stable",
            "bridge_curvature_stable",
            "analytic_bridge_derivative_pass",
            "c1_slope_match",
            "endpoint_direction_compatibility_pass",
        ):
            assert key in diag


def test_vertical_translation_invariance():
    f, base, masked = _synthetic()
    out1, d1 = adaptive_masked_handoff(f, base, masked)
    out2, d2 = adaptive_masked_handoff(f, base + 7.0, masked + 7.0)
    assert d1["status"] == d2["status"]
    if d1["status"] == "HANDOFF_ACCEPTED":
        assert d1["actual_handoff_hz"] == pytest.approx(d2["actual_handoff_hz"])
    assert np.allclose(out2, out1 + 7.0)


def test_positive_scale_invariance():
    f, base, masked = _synthetic()
    out1, d1 = adaptive_masked_handoff(f, base, masked)
    out2, d2 = adaptive_masked_handoff(f, base * 2.0, masked * 2.0)
    assert d1["status"] == d2["status"]
    if d1["status"] == "HANDOFF_ACCEPTED":
        assert d1["actual_handoff_hz"] == pytest.approx(d2["actual_handoff_hz"])
    assert np.allclose(out2, out1 * 2.0)


def test_exact_anchor_is_required():
    f = np.geomspace(900.0, 5000.0, 80)
    base = np.zeros_like(f)
    masked = np.ones_like(f)
    out, diag = adaptive_masked_handoff(f, base, masked)
    assert diag["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(out, base)
