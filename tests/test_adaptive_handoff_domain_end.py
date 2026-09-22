import numpy as np
import pytest

from engine.adaptive_handoff import adaptive_masked_handoff, _evaluate_exact_c1_bridge


def _grid():
    return np.geomspace(500.0, 5000.0, 161)


def _synthetic():
    f = _grid()
    x = np.log2(f / 1000.0)
    base = 0.8 * x
    masked = 1.8 * x + 0.15 * np.tanh(2.0 * x)
    return f, base, masked


def test_locked_bounds_are_enforced():
    f, base, masked = _synthetic()
    with pytest.raises(ValueError):
        adaptive_masked_handoff(f, base, masked, min_transition_octaves=0.32)
    with pytest.raises(ValueError):
        adaptive_masked_handoff(f, base, masked, max_transition_octaves=0.81)


def test_nominal_hz_and_domain_end_hz_compatibility():
    f, base, masked = _synthetic()
    out, diag = adaptive_masked_handoff(
        f, base, masked, nominal_hz=1000.0, domain_end_hz=3000.0
    )
    assert diag["nominal_anchor_hz"] == 1000.0
    assert len(out) == len(f)
    if diag["status"] == "HANDOFF_ACCEPTED":
        assert diag["actual_handoff_hz"] <= 3000.0 + 1e-9


def test_domain_end_hz_is_a_candidate_ceiling_not_a_grid_truncation():
    f, base, masked = _synthetic()
    out, diag = adaptive_masked_handoff(
        f, base, masked, domain_end_hz=1200.0
    )
    assert len(out) == len(f)
    assert diag["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(out, base)


def test_exact_c1_retains_raw_endpoint_slopes():
    values, diag = _evaluate_exact_c1_bridge(0.0, 1.0, 1.0, 1.0, 1.0)
    assert np.isclose(diag["used_start_slope"], 1.0)
    assert np.isclose(diag["used_end_slope"], 1.0)
    assert diag["bridge_pass"] is True
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_alpha_beta_hard_gate():
    with pytest.raises(ValueError):
        _evaluate_exact_c1_bridge(0.0, 1.0, 3.5, 3.5, 1.0)


def test_grid_below_lower_bound_fails_safe():
    f = np.geomspace(1000.0, 1200.0, 40)
    x = np.log2(f / 1000.0)
    base = 0.5 * x
    masked = 0.7 * x + 0.1 * x * x
    out, diag = adaptive_masked_handoff(f, base, masked)
    assert diag["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(out, base)


def test_no_stable_handoff_fails_safe_to_base():
    f = _grid()
    base = np.zeros_like(f)
    x = np.log2(f / 1000.0)
    masked = 2.0 - 2.0 * x
    out, diag = adaptive_masked_handoff(f, base, masked)
    assert diag["status"] == "NO_STABLE_HANDOFF"
    assert np.array_equal(out, base)


def test_diagnostics_schema_consistency():
    f, base, masked = _synthetic()
    _, diag = adaptive_masked_handoff(f, base, masked)
    assert {"status", "selection_rule", "domain_end_hz"}.issubset(diag)
    assert diag["domain_end_hz"] is None
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


def test_output_preserves_mask_after_e_and_base_before_h():
    f, base, masked = _synthetic()
    out, diag = adaptive_masked_handoff(f, base, masked)
    if diag["status"] != "HANDOFF_ACCEPTED":
        pytest.skip("synthetic candidate did not pass all locked hard gates")
    e = np.searchsorted(f, diag["actual_handoff_hz"])
    h = np.flatnonzero(np.isclose(f, 1000.0))[0]
    assert np.array_equal(out[:h + 1], base[:h + 1])
    assert np.array_equal(out[e + 1:], masked[e + 1:])
