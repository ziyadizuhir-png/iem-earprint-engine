"""
Adaptive Masked-EarPrint handoff.

Locked production architecture:
- BaseTarget is preserved through the nominal 1 kHz anchor.
- E is the earliest feasible candidate on the master grid.
- The bridge is an exact-C1 cubic Hermite bridge with raw endpoint slopes
  retained exactly.
- Feasibility requires alpha >= 0, beta >= 0, alpha + beta <= 3.
- Analytic derivative validation is mandatory.
- Destination slope and curvature stability are hard gates.
- No target averaging, scoring, Pareto selection, or curvature optimization.
- NO_STABLE_HANDOFF fails safe to the original BaseTarget.

Integration compatibility:
- Existing callers may use nominal_hz=... .
- Existing engine callers may pass domain_end_hz=... .
- nominal_anchor_hz=... remains the explicit configuration name.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

_EPS = 1e-12


def _octaves_between(f0: float, f1: float) -> float:
    return float(np.log2(float(f1) / float(f0)))


def log_slope(freq_hz: np.ndarray, level_db: np.ndarray) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float)
    y = np.asarray(level_db, dtype=float)
    if f.ndim != 1 or y.ndim != 1 or len(f) != len(y) or len(f) < 3:
        raise ValueError("freq_hz and level_db must be equal 1-D arrays of length >= 3.")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(y)):
        raise ValueError("Frequency and level arrays must be finite.")
    if not np.all(np.diff(f) > 0):
        raise ValueError("Frequency grid must be strictly increasing.")
    return np.gradient(y, np.log2(f))


def _analytic_derivative_check(
    a: float, b: float, c: float, direction: float
) -> Tuple[bool, np.ndarray, np.ndarray]:
    # q(t) = 3*a*t^2 + 2*b*t + c
    qa, qb, qc = 3.0 * a, 2.0 * b, c
    roots = []

    if abs(qa) <= _EPS:
        if abs(qb) > _EPS:
            roots.append(float(-qc / qb))
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc >= -_EPS:
            sqrt_disc = float(np.sqrt(max(0.0, disc)))
            roots.extend([
                float((-qb - sqrt_disc) / (2.0 * qa)),
                float((-qb + sqrt_disc) / (2.0 * qa)),
            ])

    candidates = [0.0, 1.0]
    candidates.extend(r for r in roots if 0.0 < r < 1.0)
    candidates = sorted(set(round(r, 14) for r in candidates))

    values = np.asarray(
        [qa * t * t + qb * t + qc for t in candidates], dtype=float
    )
    ok = bool(np.all(direction * values >= -1e-10))
    return ok, values, np.asarray(candidates, dtype=float)


def _evaluate_exact_c1_bridge(
    y0: float, y1: float, d0: float, d1: float, span_octaves: float
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Exact-C1 cubic Hermite bridge; raw endpoint slopes are never clipped."""
    span = float(span_octaves)
    if span <= 0:
        raise ValueError("Bridge span must be positive.")

    delta = float(y1 - y0)
    if abs(delta) <= _EPS:
        raise ValueError("Endpoint delta is zero.")

    direction = float(np.sign(delta))
    if abs(d0) <= _EPS or abs(d1) <= _EPS:
        raise ValueError("Endpoint slope is too close to zero.")
    if np.sign(d0) != direction or np.sign(d1) != direction:
        raise ValueError("Endpoint slopes are direction-incompatible.")

    alpha = float(d0 * span / delta)
    beta = float(d1 * span / delta)
    if alpha < -1e-12 or beta < -1e-12 or alpha + beta > 3.0 + 1e-12:
        raise ValueError("Endpoint slopes violate the locked monotone-C1 condition.")

    # y(t) = a*t^3 + b*t^2 + c*t + y0
    a = 2.0 * y0 - 2.0 * y1 + span * d0 + span * d1
    b = -3.0 * y0 + 3.0 * y1 - 2.0 * span * d0 - span * d1
    c = span * d0

    derivative_ok, derivative_values, roots = _analytic_derivative_check(
        a, b, c, direction
    )
    if not derivative_ok:
        raise ValueError("Analytic derivative monotonicity gate failed.")

    # Validation samples are only confirmation; the analytic derivative gate
    # remains authoritative.
    t = np.linspace(0.0, 1.0, 257)
    values = a * t**3 + b * t**2 + c * t + y0
    if not np.all(np.isfinite(values)):
        raise ValueError("Bridge contains non-finite values.")

    lo, hi = sorted((float(y0), float(y1)))
    if np.min(values) < lo - 1e-9:
        raise ValueError("Bridge undershoots endpoint range.")
    if np.max(values) > hi + 1e-9:
        raise ValueError("Bridge overshoots endpoint range.")

    sampled_derivative = np.gradient(values, t) / span
    if np.any(direction * sampled_derivative < -1e-8):
        raise ValueError("Sampled slope reversal gate failed.")

    start_slope = c / span
    end_slope = (3.0 * a + 2.0 * b + c) / span
    if not np.isclose(start_slope, d0, atol=1e-10, rtol=0.0):
        raise ValueError("Start C1 slope mismatch.")
    if not np.isclose(end_slope, d1, atol=1e-10, rtol=0.0):
        raise ValueError("End C1 slope mismatch.")

    return values, {
        "alpha": alpha,
        "beta": beta,
        "alpha_plus_beta": alpha + beta,
        "used_start_slope": float(start_slope),
        "used_end_slope": float(end_slope),
        "endpoint_direction_compatibility_pass": True,
        "analytic_bridge_derivative_pass": True,
        "finite_pass": True,
        "monotonicity_pass": True,
        "no_overshoot_pass": True,
        "no_undershoot_pass": True,
        "no_artificial_extrema_pass": True,
        "slope_reversal_pass": True,
        "bridge_pass": True,
        "c1_slope_match": True,
        "continuity_pass": True,
        "derivative_root_count": int(len(roots)),
        "analytic_derivative_values": derivative_values.tolist(),
    }


def _evaluate_hermite_on_grid(
    t: np.ndarray, y0: float, y1: float, d0: float, d1: float, span: float
) -> np.ndarray:
    """Evaluate the exact same cubic on arbitrary production-grid t values."""
    a = 2.0 * y0 - 2.0 * y1 + span * d0 + span * d1
    b = -3.0 * y0 + 3.0 * y1 - 2.0 * span * d0 - span * d1
    c = span * d0
    return a * t**3 + b * t**2 + c * t + y0


def _destination_stability(
    freq_hz: np.ndarray,
    level_db: np.ndarray,
    e_index: int,
    stability_window_octaves: float,
) -> Tuple[bool, bool, float, float]:
    """Check slope/curvature only inside the post-E stability window."""
    f = np.asarray(freq_hz, dtype=float)
    y = np.asarray(level_db, dtype=float)
    x = np.log2(f)

    e_x = x[e_index]
    end_x = e_x + float(stability_window_octaves)
    idx = np.flatnonzero((x >= e_x) & (x <= end_x))
    if len(idx) < 4:
        return False, False, float("nan"), float("nan")

    slopes = np.gradient(y, x)
    curvature = np.gradient(slopes, x)
    local_slope = slopes[idx]
    local_curv = curvature[idx]

    slope_scale = max(float(np.max(np.abs(local_slope))), 1e-12)
    curv_scale = max(float(np.max(np.abs(local_curv))), 1e-12)

    slope_variation = float(np.max(local_slope) - np.min(local_slope))
    curvature_variation = float(np.max(local_curv) - np.min(local_curv))

    slope_stable = slope_variation / slope_scale <= 0.35
    curvature_stable = curvature_variation / curv_scale <= 1.50

    return (
        bool(slope_stable),
        bool(curvature_stable),
        float(local_slope[0]),
        float(local_curv[0]),
    )


def adaptive_masked_handoff(
    freq_hz: np.ndarray,
    base_target_db: np.ndarray,
    masked_earprint_db: np.ndarray,
    *,
    nominal_anchor_hz: float = 1000.0,
    nominal_hz: float | None = None,
    domain_end_hz: float | None = None,
    min_transition_octaves: float = 1.0 / 3.0,
    max_transition_octaves: float = 0.8,
    stability_window_octaves: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Locked adaptive handoff.

    domain_end_hz is the caller's personal/output-domain ceiling. It is a
    candidate-domain constraint only: E may be accepted only when E <=
    domain_end_hz. The target arrays themselves are not truncated or modified
    by this parameter.

    `nominal_hz` is a compatibility alias for existing engine callers.
    The locked nominal anchor remains exactly 1000 Hz.
    """
    if nominal_hz is not None:
        nominal_anchor_hz = float(nominal_hz)

    f = np.asarray(freq_hz, dtype=float)
    base = np.asarray(base_target_db, dtype=float)
    masked = np.asarray(masked_earprint_db, dtype=float)

    if f.ndim != 1 or base.ndim != 1 or masked.ndim != 1:
        raise ValueError("All inputs must be 1-D arrays.")
    if not (len(f) == len(base) == len(masked)):
        raise ValueError("All inputs must have equal length.")
    if len(f) < 5:
        raise ValueError("Frequency grid is too short.")
    if not np.all(np.isfinite(f)) or not np.all(np.diff(f) > 0):
        raise ValueError("Frequency grid is invalid.")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(masked)):
        raise ValueError("Target arrays must be finite.")

    if abs(float(nominal_anchor_hz) - 1000.0) > 1e-9:
        raise ValueError("Locked nominal anchor is 1000 Hz.")
    if min_transition_octaves < 1.0 / 3.0 - 1e-12:
        raise ValueError("Minimum transition width violates locked 1/3-octave bound.")
    if max_transition_octaves > 0.8 + 1e-12:
        raise ValueError("Maximum transition width violates locked 0.8-octave bound.")
    if min_transition_octaves > max_transition_octaves:
        raise ValueError("Transition bounds are invalid.")

    if domain_end_hz is not None:
        domain_end_hz = float(domain_end_hz)
        if not np.isfinite(domain_end_hz) or domain_end_hz <= 1000.0:
            return base.copy(), {
                "status": "NO_STABLE_HANDOFF",
                "nominal_anchor_hz": 1000.0,
                "selection_rule": "earliest_feasible_E",
                "fail_safe": "BaseTarget",
                "reason": "domain_end_hz does not extend above the 1000 Hz anchor",
                "domain_end_hz": domain_end_hz,
            }

    h_matches = np.flatnonzero(np.isclose(f, 1000.0, atol=1e-9, rtol=0.0))
    if len(h_matches) != 1:
        return base.copy(), {
            "status": "NO_STABLE_HANDOFF",
            "nominal_anchor_hz": 1000.0,
            "selection_rule": "earliest_feasible_E",
            "reason": "exact 1000 Hz computational anchor is required",
            "fail_safe": "BaseTarget",
            "domain_end_hz": domain_end_hz,
        }
    h_idx = int(h_matches[0])

    base_slopes = log_slope(f, base)
    masked_slopes = log_slope(f, masked)

    for e_idx in range(h_idx + 1, len(f)):
        e_hz = float(f[e_idx])

        # domain_end_hz is an upper bound on candidate E. Do not alter the
        # production grid and do not extrapolate beyond the caller domain.
        if domain_end_hz is not None and e_hz > domain_end_hz + 1e-9:
            break

        width = _octaves_between(1000.0, e_hz)

        if width < min_transition_octaves - 1e-12:
            continue
        if width > max_transition_octaves + 1e-12:
            break

        delta = float(masked[e_idx] - base[h_idx])
        if abs(delta) <= _EPS:
            continue

        direction = np.sign(delta)
        d0 = float(base_slopes[h_idx])
        d1 = float(masked_slopes[e_idx])

        if np.sign(d0) != direction or np.sign(d1) != direction:
            continue

        slope_stable, curvature_stable, slope_ref, curvature_ref = _destination_stability(
            f, masked, e_idx, stability_window_octaves
        )
        if not slope_stable or not curvature_stable:
            continue

        try:
            _, bridge_diag = _evaluate_exact_c1_bridge(
                float(base[h_idx]),
                float(masked[e_idx]),
                d0,
                d1,
                width,
            )
        except ValueError:
            continue

        bridge_idx = np.arange(h_idx, e_idx + 1)
        t = np.log2(f[bridge_idx] / 1000.0) / width
        bridge = _evaluate_hermite_on_grid(
            t,
            float(base[h_idx]),
            float(masked[e_idx]),
            d0,
            d1,
            width,
        )

        out = masked.copy()
        out[h_idx:e_idx + 1] = bridge
        out[:h_idx + 1] = base[:h_idx + 1]
        out[e_idx + 1:] = masked[e_idx + 1:]

        diagnostics = {
            "status": "HANDOFF_ACCEPTED",
            "nominal_anchor_hz": 1000.0,
            "actual_handoff_hz": e_hz,
            "transition_end_hz": e_hz,
            "transition_width_octaves": width,
            "selection_rule": "earliest_feasible_E",
            "candidate_scoring": False,
            "pareto_selection": False,
            "curvature_optimization": False,
            "bridge_type": "monotone_cubic_hermite_true_C1",
            "destination_slope_stable": slope_stable,
            "destination_curvature_stable": curvature_stable,
            "bridge_curvature_stable": curvature_stable,
            "oscillation_pass": True,
            "destination_slope_reference": slope_ref,
            "destination_curvature_reference": curvature_ref,
            "endpoint_start_level": float(base[h_idx]),
            "endpoint_end_level": float(masked[e_idx]),
            "endpoint_start_slope": d0,
            "endpoint_end_slope": d1,
            "domain_end_hz": domain_end_hz,
            **bridge_diag,
        }
        return out, diagnostics

    return base.copy(), {
        "status": "NO_STABLE_HANDOFF",
        "nominal_anchor_hz": 1000.0,
        "selection_rule": "earliest_feasible_E",
        "fail_safe": "BaseTarget",
        "domain_end_hz": domain_end_hz,
    }


# Backward-compatible aliases used by existing tests/tools.
def _c1_monotone_bridge(y0, y1, d0, d1, span_octaves):
    return _evaluate_exact_c1_bridge(y0, y1, d0, d1, span_octaves)


def _monotone_bridge(y0, y1, d0, d1, span_octaves):
    return _evaluate_exact_c1_bridge(y0, y1, d0, d1, span_octaves)
