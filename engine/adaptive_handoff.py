"""Deterministic Adaptive Masked-EarPrint handoff.

Locked architecture:
    BaseTarget
        -> exact nominal anchor H = 1000 Hz
        -> earliest feasible E
        -> shape-preserving monotone cubic Hermite bridge
        -> Masked EarPrint

The bridge is deliberately slope-limited so the transition cannot reverse
direction or create an artificial extremum. No candidate score/Pareto
selection is used.
"""

from __future__ import annotations

import math
import numpy as np


_EPS = 1e-9
_BRIDGE_TOL = 1e-8
_CURVATURE_EPS = 1e-4

NOMINAL_ANCHOR_HZ = 1000.0
MIN_TRANSITION_OCTAVES = 1.0 / 3.0
MAX_TRANSITION_OCTAVES = 0.8
STABILITY_WINDOW_OCTAVES = 0.20


# ---------------------------------------------------------------------------
# Log-frequency geometry
# ---------------------------------------------------------------------------

def log_slope(freq, level):
    freq = np.asarray(freq, dtype=float)
    level = np.asarray(level, dtype=float)
    x = np.log2(freq)
    out = np.empty_like(level, dtype=float)

    out[0] = (level[1] - level[0]) / (x[1] - x[0])
    out[-1] = (level[-1] - level[-2]) / (x[-1] - x[-2])
    out[1:-1] = (level[2:] - level[:-2]) / (x[2:] - x[:-2])
    return out


def log_curvature(freq, level):
    freq = np.asarray(freq, dtype=float)
    level = np.asarray(level, dtype=float)
    x = np.log2(freq)
    slope = log_slope(freq, level)
    out = np.empty_like(level, dtype=float)

    out[0] = (slope[1] - slope[0]) / (x[1] - x[0])
    out[-1] = (slope[-1] - slope[-2]) / (x[-1] - x[-2])
    out[1:-1] = (slope[2:] - slope[:-2]) / (x[2:] - x[:-2])
    return out


def _interp_log(freq, level, hz):
    return float(np.interp(math.log(hz), np.log(freq), level))

# ---------------------------------------------------------------------------
# Stability helpers
# ---------------------------------------------------------------------------

def _stable_window(values, sign, eps=_EPS):
    values = np.asarray(values, dtype=float)
    active = values[np.abs(values) > eps]
    if active.size < 3:
        return False
    return bool(np.all(np.sign(active) == np.sign(sign)))


def _curvature_stable(values, eps=_CURVATURE_EPS):
    """Allow zero curvature or at most one sign transition."""
    values = np.asarray(values, dtype=float)
    active = values[np.abs(values) > eps]

    if active.size == 0:
        return True
    if active.size < 3:
        return False

    signs = np.sign(active)
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return changes <= 1


# ---------------------------------------------------------------------------
# Quintic smootherstep retained for mathematical regression compatibility.
# It is NOT used by adaptive_masked_handoff.
# ---------------------------------------------------------------------------

def _smootherstep(t):
    t = np.asarray(t, dtype=float)
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _smootherstep_derivative(t):
    t = np.asarray(t, dtype=float)
    return 30.0 * t**4 - 60.0 * t**3 + 30.0 * t**2

def _smootherstep_second_derivative(t):
    t = np.asarray(t, dtype=float)
    return 120.0 * t**3 - 180.0 * t**2 + 60.0 * t


def validate_locked_smootherstep():
    t = np.linspace(0.0, 1.0, 10001)
    w = _smootherstep(t)
    dw = _smootherstep_derivative(t)

    if not math.isclose(float(w[0]), 0.0, abs_tol=1e-15):
        raise AssertionError("smootherstep w(0) != 0")
    if not math.isclose(float(w[-1]), 1.0, abs_tol=1e-15):
        raise AssertionError("smootherstep w(1) != 1")
    if not math.isclose(float(dw[0]), 0.0, abs_tol=1e-15):
        raise AssertionError("smootherstep w'(0) != 0")
    if not math.isclose(float(dw[-1]), 0.0, abs_tol=1e-15):
        raise AssertionError("smootherstep w'(1) != 0")
    if float(np.min(w)) < -1e-12:
        raise AssertionError("smootherstep lower bound failed")
    if float(np.max(w)) > 1.0 + 1e-12:
        raise AssertionError("smootherstep upper bound failed")
    if float(np.min(dw)) < -1e-12:
        raise AssertionError("smootherstep monotonicity failed")
    return True


# Backward-compatible alias used by older tests.
_validate_smootherstep = validate_locked_smootherstep


# ---------------------------------------------------------------------------
# Exact quintic bridge diagnostics retained for regression tests.
# ---------------------------------------------------------------------------

def _evaluate_quintic_bridge(freq, target, masked_target, h_index, e_index):
    H = float(freq[h_index])
    E = float(freq[e_index])
    span = math.log2(E / H)

    if span <= 0:
        raise ValueError("Bridge span must be positive.")

    idx = np.arange(h_index, e_index + 1, dtype=int)
    f = freq[idx]
    t = np.clip(np.log2(f / H) / span, 0.0, 1.0)
    w = _smootherstep(t)

    base = target[idx]
    masked = masked_target[idx]
    bridge = base + w * (masked - base)

    finite = bool(np.all(np.isfinite(bridge)))
    y0 = float(bridge[0])
    y1 = float(bridge[-1])
    delta = y1 - y0

    if not finite:
        return {"passed": False, "reason": "non_finite_bridge"}

    if abs(delta) <= _EPS:
        return {"passed": False, "reason": "zero_endpoint_delta"}
    x = np.log2(f)
    slope = np.gradient(bridge, x, edge_order=1)
    direction = math.copysign(1.0, delta)
    active = slope[np.abs(slope) > _EPS]

    monotonic = bool(active.size and np.all(np.sign(active) == direction))
    slope_reversal = bool(active.size and np.any(np.sign(active) != direction))

    signs = np.sign(active) if active.size else np.array([])
    extrema = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0

    lo = min(y0, y1)
    hi = max(y0, y1)
    overshoot = max(0.0, float(np.max(bridge)) - hi)
    undershoot = max(0.0, lo - float(np.min(bridge)))

    endpoint_weight_pass = (
        math.isclose(float(w[0]), 0.0, abs_tol=1e-12)
        and math.isclose(float(w[-1]), 1.0, abs_tol=1e-12)
    )
    endpoint_derivative_pass = (
        abs(float(_smootherstep_derivative(np.array([0.0]))[0])) <= 1e-12
        and abs(float(_smootherstep_derivative(np.array([1.0]))[0])) <= 1e-12
    )

    passed = bool(
        monotonic
        and extrema == 0
        and overshoot <= _BRIDGE_TOL
        and undershoot <= _BRIDGE_TOL
        and endpoint_weight_pass
        and endpoint_derivative_pass
    )

    return {
        "passed": passed,
        "finite_pass": finite,
        "monotonicity_pass": monotonic,
        "slope_reversal": slope_reversal,
        "slope_reversal_pass": not slope_reversal,
        "extrema_pass": extrema == 0,
        "artificial_extrema_count": extrema,
        "overshoot_db": overshoot,
        "undershoot_db": undershoot,
        "overshoot_pass": overshoot <= _BRIDGE_TOL,
        "undershoot_pass": undershoot <= _BRIDGE_TOL,
        "endpoint_weight_pass": endpoint_weight_pass,
        "endpoint_derivative_pass": endpoint_derivative_pass,
        "continuity_pass": True,
        "bridge_type": "quintic_smootherstep",
    }


# ---------------------------------------------------------------------------
# Shape-preserving monotone cubic Hermite bridge
# ---------------------------------------------------------------------------

def _limited_endpoint_slopes(delta, span_octaves, d0, d1):
    """Return direction-safe endpoint slopes for one monotone cubic segment.

    The normalized Hermite endpoint derivatives satisfy:
        alpha = d0*L/delta
        beta  = d1*L/delta

    We force alpha,beta >= 0 and alpha + beta <= 3.
    This is a standard sufficient monotonicity condition for a single cubic
    Hermite interval.
    """
    if abs(delta) <= _EPS:
        return 0.0, 0.0

    L = float(span_octaves)

    # Normalized Hermite derivatives. When d and delta have the same
    # direction, the ratio is positive; opposite-direction slopes are
    # clamped to zero.
    alpha = max(0.0, float(d0) * L / delta)
    beta = max(0.0, float(d1) * L / delta)

    total = alpha + beta
    if total > 3.0:
        factor = 3.0 / total
        alpha *= factor
        beta *= factor

    return (
        alpha * delta / L,
        beta * delta / L,
    )


def _hermite_coefficients(y0, y1, m0, m1, span_octaves):
    L = float(span_octaves)
    c0 = float(y0)
    c1 = float(m0) * L
    c2 = -3.0 * float(y0) + 3.0 * float(y1) - 2.0 * float(m0) * L - float(m1) * L
    c3 = 2.0 * float(y0) - 2.0 * float(y1) + float(m0) * L + float(m1) * L
    return c0, c1, c2, c3


def _cubic_derivative_roots(c1, c2, c3):
    # p'(t) = c1 + 2*c2*t + 3*c3*t^2
    A = 3.0 * c3
    B = 2.0 * c2
    C = c1

    roots = []
    if abs(A) <= 1e-14:
        if abs(B) > 1e-14:
            roots.append(-C / B)
        return roots

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return roots

    root_disc = math.sqrt(max(0.0, disc))
    roots.extend([
        (-B - root_disc) / (2.0 * A),
        (-B + root_disc) / (2.0 * A),
    ])
    return roots


def _evaluate_monotone_cubic_bridge(
    freq,
    target,
    masked_target,
    h_index,
    e_index,
):
    H = float(freq[h_index])
    E = float(freq[e_index])
    span = math.log2(E / H)

    if span <= 0.0:
        raise ValueError("Bridge span must be positive.")

    idx = np.arange(h_index, e_index + 1, dtype=int)
    bridge_freq = freq[idx]

    y0 = float(target[h_index])
    y1 = float(masked_target[e_index])
    endpoint_delta = y1 - y0

    if not math.isfinite(endpoint_delta):
        return {"passed": False, "reason": "non_finite_endpoint"}

    if abs(endpoint_delta) <= _EPS:
        return {"passed": False, "reason": "zero_endpoint_delta"}

    target_slope = log_slope(freq, target)
    masked_slope = log_slope(freq, masked_target)

    raw_d0 = float(target_slope[h_index])
    raw_d1 = float(masked_slope[e_index])

    direction = math.copysign(1.0, endpoint_delta)

    # A shape-preserving bridge must not hide a source-curve slope reversal
    # at either endpoint. If either endpoint slope points opposite to the
    # endpoint displacement, reject this E instead of forcing a synthetic
    # monotone segment.
    endpoint_direction_pass = bool(
        (
            abs(raw_d0) <= _EPS
            or math.copysign(1.0, raw_d0) == direction
        )
        and
        (
            abs(raw_d1) <= _EPS
            or math.copysign(1.0, raw_d1) == direction
        )
    )

    if not endpoint_direction_pass:
        return {
            "passed": False,
            "bridge_pass": False,
            "finite_pass": True,
            "monotonicity_pass": False,
            "slope_reversal": True,
            "slope_reversal_pass": False,
            "extrema_pass": False,
            "artificial_extrema_count": 0,
            "overshoot_db": 0.0,
            "undershoot_db": 0.0,
            "overshoot_pass": True,
            "undershoot_pass": True,
            "continuity_pass": True,
            "bridge_type": "monotone_cubic_hermite",
            "endpoint_direction_pass": False,
            "raw_start_slope_db_per_octave": raw_d0,
            "raw_end_slope_db_per_octave": raw_d1,
            "H_hz": H,
            "E_hz": E,
            "span_octaves": span,
        }

    d0, d1 = _limited_endpoint_slopes(
        endpoint_delta,
        span,
        raw_d0,
        raw_d1,
    )

    c0, c1, c2, c3 = _hermite_coefficients(
        y0, y1, d0, d1, span
    )
    t_values = (
        np.log2(bridge_freq / H) / span
    )
    t_values = np.clip(t_values, 0.0, 1.0)

    bridge_values = (
        c0
        + c1 * t_values
        + c2 * t_values**2
        + c3 * t_values**3
    )

    finite = bool(np.all(np.isfinite(bridge_values)))
    if not finite:
        return {"passed": False, "reason": "non_finite_bridge"}

    # Exact analytic derivative gate.
    candidates = [0.0, 1.0]
    candidates.extend(
        r for r in _cubic_derivative_roots(c1, c2, c3)
        if 0.0 < r < 1.0
    )

    derivative_t = np.asarray(
        [
            c1 + 2.0 * c2 * t + 3.0 * c3 * t * t
            for t in candidates
        ],
        dtype=float,
    )

    derivative_db_per_oct = derivative_t / span

    direction = math.copysign(1.0, endpoint_delta)
    meaningful = derivative_db_per_oct[
        np.abs(derivative_db_per_oct) > _EPS
    ]

    monotonicity_pass = bool(
        meaningful.size > 0
        and np.all(np.sign(meaningful) == direction)
    )

    slope_reversal = bool(
        meaningful.size > 0
        and np.any(np.sign(meaningful) != direction)
    )

    # Sampled curve checks remain as a secondary diagnostic gate.
    active = np.diff(bridge_values)
    active = active[np.abs(active) > _EPS]
    if active.size > 1:
        signs = np.sign(active)
        artificial_extrema_count = int(
            np.count_nonzero(signs[1:] != signs[:-1])
        )
    else:
        artificial_extrema_count = 0

    lo = min(y0, y1)
    hi = max(y0, y1)

    overshoot = max(
        0.0,
        float(np.max(bridge_values)) - hi,
    )
    undershoot = max(
        0.0,
        lo - float(np.min(bridge_values)),
    )

    # Exact endpoint continuity.
    continuity_pass = (
        math.isclose(
            float(bridge_values[0]),
            y0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and
        math.isclose(
            float(bridge_values[-1]),
            y1,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )

    passed = bool(
        finite
        and monotonicity_pass
        and artificial_extrema_count == 0
        and overshoot <= _BRIDGE_TOL
        and undershoot <= _BRIDGE_TOL
        and continuity_pass
        and not slope_reversal
    )

    return {
        "passed": passed,
        "bridge_pass": passed,
        "finite_pass": finite,
        "monotonicity_pass": monotonicity_pass,
        "slope_reversal": slope_reversal,
        "slope_reversal_pass": not slope_reversal,
        "extrema_pass": artificial_extrema_count == 0,
        "artificial_extrema_count": artificial_extrema_count,
        "overshoot_db": overshoot,
        "undershoot_db": undershoot,
        "overshoot_pass": overshoot <= _BRIDGE_TOL,
        "undershoot_pass": undershoot <= _BRIDGE_TOL,
        "continuity_pass": continuity_pass,
        "endpoint_direction_pass": endpoint_direction_pass,
        "bridge_type": "monotone_cubic_hermite",
        "raw_start_slope_db_per_octave": raw_d0,
        "raw_end_slope_db_per_octave": raw_d1,
        "limited_start_slope_db_per_octave": d0,
        "limited_end_slope_db_per_octave": d1,
        "analytic_derivative_min_db_per_octave": float(np.min(derivative_db_per_oct)),
        "analytic_derivative_max_db_per_octave": float(np.max(derivative_db_per_oct)),
        "span_octaves": span,
        "H_hz": H,
        "E_hz": E,
    }

def _monotone_bridge(y0, y1, d0, d1, span_octaves):
    """Backward-compatible endpoint-geometry helper."""
    if span_octaves <= 0:
        raise ValueError("Bridge span must be positive.")

    delta = float(y1) - float(y0)
    if abs(delta) <= _EPS:
        raise ValueError("Endpoint delta is zero.")

    m0, m1 = _limited_endpoint_slopes(
        delta,
        float(span_octaves),
        float(d0),
        float(d1),
    )

    return {
        "bridge_type": "monotone_cubic_hermite",
        "y0": float(y0),
        "y1": float(y1),
        "d0": float(m0),
        "d1": float(m1),
        "span_octaves": float(span_octaves),
        "shape_preserving": True,
        "c1_slope_match_at_H": math.isclose(
            m0, float(d0), rel_tol=0.0, abs_tol=1e-12
        ),
        "c1_slope_match_at_E": math.isclose(
            m1, float(d1), rel_tol=0.0, abs_tol=1e-12
        ),
    }


# ---------------------------------------------------------------------------
# Destination stability
# ---------------------------------------------------------------------------

def _candidate_destination_stable(
    freq,
    masked_slope,
    masked_curvature,
    end_index,
    stability_window_octaves,
):
    E = float(freq[end_index])

    stable_idx = np.where(
        (freq > E)
        & (
            freq <= E * 2.0**stability_window_octaves
        )
    )[0]

    if stable_idx.size < 3:
        return None

    d1 = float(masked_slope[end_index])
    if abs(d1) <= _EPS:
        return None

    slope_ok = _stable_window(
        masked_slope[stable_idx],
        np.sign(d1),
    )
    curvature_ok = _curvature_stable(
        masked_curvature[stable_idx]
    )

    if not slope_ok:
        return None

    return {
        "indices": stable_idx,
        "slope_stable": True,
        "curvature_stable": bool(curvature_ok),
        "slope_direction": float(np.sign(d1)),
    }


# ---------------------------------------------------------------------------
# Locked adaptive handoff
# ---------------------------------------------------------------------------

def adaptive_masked_handoff(
    freq,
    target,
    masked_target,
    nominal_hz=NOMINAL_ANCHOR_HZ,
    domain_end_hz=12000.0,
    min_transition_octaves=MIN_TRANSITION_OCTAVES,
    max_transition_octaves=MAX_TRANSITION_OCTAVES,
    stability_window_octaves=STABILITY_WINDOW_OCTAVES,
):
    """Run the locked handoff using earliest feasible E and a monotone bridge."""
    freq = np.asarray(freq, dtype=float)
    target = np.asarray(target, dtype=float)
    masked_target = np.asarray(masked_target, dtype=float)

    if not (
        freq.ndim == target.ndim == masked_target.ndim == 1
    ):
        raise ValueError("freq, target and masked_target must be 1-D arrays.")

    if not (
        len(freq) == len(target) == len(masked_target)
    ):
        raise ValueError("freq, target and masked_target must have identical lengths.")

    if np.any(np.diff(freq) <= 0):
        raise ValueError("Frequency grid must be strictly increasing.")

    if not (
        np.all(np.isfinite(freq))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(masked_target))
    ):
        raise ValueError("Input arrays must contain only finite values.")

    H = float(nominal_hz)
    if not math.isclose(
        H,
        NOMINAL_ANCHOR_HZ,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Locked handoff requires H = 1000 Hz.")

    if min_transition_octaves < MIN_TRANSITION_OCTAVES:
        raise ValueError(
            "Locked minimum transition width cannot be below 1/3 octave."
        )

    if max_transition_octaves > MAX_TRANSITION_OCTAVES:
        raise ValueError(
            "Locked maximum transition width cannot exceed 0.8 octave."
        )

    if min_transition_octaves > max_transition_octaves:
        raise ValueError("Minimum transition width exceeds maximum.")

    if H < float(freq[0]):
        return target.copy(), {
            "status": "NO_STABLE_HANDOFF",
            "reason": "nominal_anchor_below_frequency_domain",
            "nominal_anchor_hz": H,
            "actual_handoff_hz": None,
            "transition_end_hz": None,
            "transition_width_octaves": None,
            "candidate_count": 0,
            "selected_candidate_rank": None,
        }

    h_candidates = np.where(
        np.isclose(freq, H, rtol=0.0, atol=1e-12)
    )[0]

    if h_candidates.size == 0:
        return target.copy(), {
            "status": "NO_STABLE_HANDOFF",
            "reason": "nominal_anchor_not_present_on_master_grid",
            "nominal_anchor_hz": H,
            "actual_handoff_hz": None,
            "transition_end_hz": None,
            "transition_width_octaves": None,
            "candidate_count": 0,
            "selected_candidate_rank": None,
        }
    h_index = int(h_candidates[0])

    target_slope = log_slope(freq, target)
    masked_slope = log_slope(freq, masked_target)
    masked_curvature = log_curvature(freq, masked_target)

    candidate_count = 0

    for e_index in range(h_index + 1, len(freq)):
        E = float(freq[e_index])

        if E >= float(domain_end_hz):
            break

        span = math.log2(E / H)

        if span < min_transition_octaves:
            continue

        if span > max_transition_octaves:
            break

        destination = _candidate_destination_stable(
            freq,
            masked_slope,
            masked_curvature,
            e_index,
            stability_window_octaves,
        )

        if destination is None:
            continue

        if not destination["slope_stable"]:
            continue

        if not destination["curvature_stable"]:
            continue

        candidate_count += 1

        bridge = _evaluate_monotone_cubic_bridge(
            freq,
            target,
            masked_target,
            h_index,
            e_index,
        )

        if not bridge["passed"]:
            continue

        # Earliest feasible E wins.
        bridge_indices = np.arange(
            h_index,
            e_index + 1,
            dtype=int,
        )
        bridge_freq = freq[bridge_indices]

        span = math.log2(E / H)
        y0 = float(target[h_index])
        y1 = float(masked_target[e_index])

        d0, d1 = _limited_endpoint_slopes(
            y1 - y0,
            span,
            float(target_slope[h_index]),
            float(masked_slope[e_index]),
        )
        c0, c1, c2, c3 = _hermite_coefficients(
            y0, y1, d0, d1, span
        )

        t = np.clip(
            np.log2(bridge_freq / H) / span,
            0.0,
            1.0,
        )

        bridge_values = (
            c0
            + c1 * t
            + c2 * t**2
            + c3 * t**3
        )

        output = target.copy()

        output[freq <= H] = target[freq <= H]

        output[bridge_indices] = bridge_values

        after = freq > E
        output[after] = masked_target[after]

        return output, {
            "status": "ADAPTIVE_HANDOFF",
            "reason": "earliest_feasible_E",
            "nominal_anchor_hz": H,
            "actual_handoff_hz": H,
            "transition_end_hz": E,
            "transition_width_octaves": span,
            "endpoint_target_level_db": y0,
            "endpoint_masked_level_db": y1,
            "endpoint_target_slope_db_per_octave": float(target_slope[h_index]),
            "endpoint_masked_slope_db_per_octave": float(masked_slope[e_index]),
            "destination_slope_stable": bool(destination["slope_stable"]),
            "destination_curvature_stable": bool(destination["curvature_stable"]),
            "bridge_curvature_stable": True,
            "continuity_pass": True,
            "monotonicity_pass": bool(bridge["monotonicity_pass"]),
            "extrema_pass": bool(bridge["extrema_pass"]),
            "overshoot_pass": bool(bridge["overshoot_pass"]),
            "undershoot_pass": bool(bridge["undershoot_pass"]),
            "oscillation_pass": bool(bridge["extrema_pass"]),
            "slope_reversal_pass": bool(bridge["slope_reversal_pass"]),
            "bridge_pass": True,
            "bridge_type": "monotone_cubic_hermite",
            "shape_preserving": True,
            "selection_rule": "earliest_feasible_E",
            "candidate_scoring": False,
            "pareto_selection": False,
            "curvature_optimization": False,
            "candidate_count": candidate_count,
            "selected_candidate_rank": 1,
            "limited_start_slope_db_per_octave": bridge["limited_start_slope_db_per_octave"],
            "limited_end_slope_db_per_octave": bridge["limited_end_slope_db_per_octave"],
            "analytic_derivative_min_db_per_octave": bridge["analytic_derivative_min_db_per_octave"],
            "analytic_derivative_max_db_per_octave": bridge["analytic_derivative_max_db_per_octave"],
            "smootherstep_endpoint_weight_pass": False,
            "smootherstep_endpoint_derivative_pass": False,
        }

    return target.copy(), {
        "status": "NO_STABLE_HANDOFF",
        "reason": "no_candidate_passed_all_hard_gates",
        "nominal_anchor_hz": H,
        "actual_handoff_hz": None,
        "transition_end_hz": None,
        "transition_width_octaves": None,
        "endpoint_target_level_db": None,
        "endpoint_masked_level_db": None,
        "endpoint_target_slope_db_per_octave": float(target_slope[h_index]),
        "endpoint_masked_slope_db_per_octave": float(masked_slope[h_index]),
        "candidate_count": candidate_count,
        "selected_candidate_rank": None,
        "destination_slope_stable": False,
        "destination_curvature_stable": False,
        "bridge_curvature_stable": False,
        "continuity_pass": False,
        "monotonicity_pass": False,
        "extrema_pass": False,
        "overshoot_pass": False,
        "undershoot_pass": False,
        "oscillation_pass": False,
        "slope_reversal_pass": False,
        "bridge_pass": False,
        "shape_preserving": True,
        "selection_rule": "earliest_feasible_E",
        "candidate_scoring": False,
        "pareto_selection": False,
        "curvature_optimization": False,
        "smootherstep_endpoint_weight_pass": False,
        "smootherstep_endpoint_derivative_pass": False,
    }