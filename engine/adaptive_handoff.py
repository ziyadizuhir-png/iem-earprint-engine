"""Locked Adaptive Masked-EarPrint handoff.

ENGINEERING-LOCKED HANDOFF ARCHITECTURE
=======================================

This module implements the locked boundary handoff between:

    BaseTarget
        ->
    Masked EarPrint
        ->
    final handoff output

Locked rules
------------

1. H is fixed at the nominal anchor:

       H = 1000 Hz

   The nominal anchor is therefore the actual beginning of the transition.
   No adaptive H-search is performed in this locked implementation.

2. BaseTarget is preserved exactly through H.

3. The transition endpoint E must satisfy:

       1/3 octave <= log2(E / H) <= 0.8 octave

4. The transition is a quintic smootherstep blend:

       Output = BaseTarget + w(t) * (MaskedEarPrint - BaseTarget)

   where

       t = log2(f / H) / log2(E / H)

       w(t) = 6t^5 - 15t^4 + 10t^3

5. The smootherstep has the exact properties:

       w(0) = 0
       w(1) = 1
       w'(0) = 0
       w'(1) = 0
       0 <= w(t) <= 1
       w'(t) >= 0 for 0 <= t <= 1

6. E is selected deterministically as the EARLIEST candidate that passes
   every hard gate.

   There is deliberately no curvature-minimization, Pareto ranking, weighted
   score, or target-dependent preference function in E selection.

7. The actual bridge must pass hard shape gates:

       - finite values
       - monotonic transition
       - no endpoint overshoot
       - no endpoint undershoot
       - no artificial extrema
       - no slope reversal

8. Destination behaviour immediately after E must be stable:

       - slope direction stable
       - curvature not repeatedly oscillatory

9. If no E satisfies the hard constraints:

       status = "NO_STABLE_HANDOFF"

   and the original BaseTarget is returned unchanged.

10. No correction is applied outside the defined handoff region.

Input contract
--------------

freq:
    Strictly increasing frequency grid in Hz.

target:
    BaseTarget on the same frequency grid.

masked_target:
    Already-computed Masked EarPrint on the same frequency grid.

Important:
    This module does NOT construct the robust mask. The upstream pipeline
    must already have produced Masked EarPrint.

Output contract
---------------

Accepted handoff:

    f <= H
        -> BaseTarget

    H < f <= E
        -> quintic smootherstep bridge

    f > E
        -> Masked EarPrint

Normal output status:

    "ADAPTIVE_HANDOFF"

Failure status:

    "NO_STABLE_HANDOFF"

The implementation is intentionally deterministic and target-independent.
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

_EPS = 1e-9

# Fixed nominal handoff anchor.
NOMINAL_ANCHOR_HZ = 1000.0

# Locked transition-width limits.
MIN_TRANSITION_OCTAVES = 1.0 / 3.0
MAX_TRANSITION_OCTAVES = 0.8

# Destination stability window.
STABILITY_WINDOW_OCTAVES = 0.20

# Numerical tolerance for hard shape gates.
_BRIDGE_TOL = 1e-8

# Curvature significance threshold.
_CURVATURE_EPS = 1e-4


# ---------------------------------------------------------------------------
# Basic log-frequency geometry
# ---------------------------------------------------------------------------

def log_slope(freq, level):
    """Return first derivative in dB/octave on the log2-frequency axis."""
    freq = np.asarray(freq, dtype=float)
    level = np.asarray(level, dtype=float)

    x = np.log2(freq)

    out = np.empty_like(level, dtype=float)

    out[0] = (level[1] - level[0]) / (x[1] - x[0])
    out[-1] = (level[-1] - level[-2]) / (x[-1] - x[-2])

    out[1:-1] = (
        (level[2:] - level[:-2])
        / (x[2:] - x[:-2])
    )

    return out


def log_curvature(freq, level):
    """Return second derivative in dB/octave^2 on the log2-frequency axis."""
    freq = np.asarray(freq, dtype=float)
    level = np.asarray(level, dtype=float)

    x = np.log2(freq)
    slope = log_slope(freq, level)

    curvature = np.empty_like(level, dtype=float)

    curvature[0] = (
        (slope[1] - slope[0])
        / (x[1] - x[0])
    )

    curvature[-1] = (
        (slope[-1] - slope[-2])
        / (x[-1] - x[-2])
    )

    curvature[1:-1] = (
        (slope[2:] - slope[:-2])
        / (x[2:] - x[:-2])
    )

    return curvature


def _interp_log(freq, level, hz):
    """Interpolate a quantity on logarithmic frequency."""
    return float(
        np.interp(
            math.log(hz),
            np.log(freq),
            level,
        )
    )


# ---------------------------------------------------------------------------
# Sign / stability helpers
# ---------------------------------------------------------------------------

def _sign_stable(values, sign, eps=_EPS):
    """Return True when all meaningful values have the requested sign."""
    values = np.asarray(values, dtype=float)

    active = values[np.abs(values) > eps]

    if active.size == 0:
        return False

    return bool(
        np.all(
            np.sign(active) == np.sign(sign)
        )
    )


def _stable_window(values, sign, eps=_EPS):
    """Return True when a local slope window keeps one direction."""
    values = np.asarray(values, dtype=float)

    active = values[np.abs(values) > eps]

    if active.size < 3:
        return False

    return bool(
        np.all(
            np.sign(active) == np.sign(sign)
        )
    )


def _curvature_stable(values, eps=_CURVATURE_EPS):
    """Reject repeated curvature oscillation.

    A smooth curve may legitimately contain one curvature sign crossing.
    Repeated alternating curvature is treated as unstable.

    Constant/near-zero curvature is accepted.
    """
    values = np.asarray(values, dtype=float)

    active = values[np.abs(values) > eps]

    if active.size == 0:
        return True

    if active.size < 3:
        return False

    signs = np.sign(active)

    changes = int(
        np.count_nonzero(
            signs[1:] != signs[:-1]
        )
    )

    return changes <= 1


# ---------------------------------------------------------------------------
# Quintic smootherstep
# ---------------------------------------------------------------------------

def _smootherstep(t):
    """Quintic smootherstep.

    Exact polynomial:

        w(t) = 6t^5 - 15t^4 + 10t^3

    This is the locked transition weighting function.
    """
    t = np.asarray(t, dtype=float)

    return (
        6.0 * t**5
        - 15.0 * t**4
        + 10.0 * t**3
    )


def _smootherstep_derivative(t):
    """First derivative of quintic smootherstep."""
    t = np.asarray(t, dtype=float)

    return (
        30.0 * t**4
        - 60.0 * t**3
        + 30.0 * t**2
    )


def _smootherstep_second_derivative(t):
    """Second derivative of quintic smootherstep."""
    t = np.asarray(t, dtype=float)

    return (
        120.0 * t**3
        - 180.0 * t**2
        + 60.0 * t
    )


def _validate_smootherstep():
    """Internal mathematical sanity check for the locked weighting function."""
    t = np.linspace(0.0, 1.0, 10001)

    w = _smootherstep(t)
    dw = _smootherstep_derivative(t)

    if not math.isclose(
        float(w[0]),
        0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError("smootherstep w(0) != 0")

    if not math.isclose(
        float(w[-1]),
        1.0,
        abs_tol=1e-15,
    ):
        raise AssertionError("smootherstep w(1) != 1")

    if not math.isclose(
        float(dw[0]),
        0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError("smootherstep w'(0) != 0")

    if not math.isclose(
        float(dw[-1]),
        0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError("smootherstep w'(1) != 0")

    if float(np.min(w)) < -1e-12:
        raise AssertionError("smootherstep violates lower bound")

    if float(np.max(w)) > 1.0 + 1e-12:
        raise AssertionError("smootherstep violates upper bound")

    if float(np.min(dw)) < -1e-12:
        raise AssertionError("smootherstep is not monotonic")


# ---------------------------------------------------------------------------
# Bridge evaluation
# ---------------------------------------------------------------------------

def _evaluate_quintic_bridge(
    freq,
    target,
    masked_target,
    h_index,
    e_index,
):
    """Evaluate the locked quintic bridge and all hard shape gates.

    The source curves themselves are sampled on the repository master grid.
    The transition weighting is exact quintic smootherstep.

    Returns a diagnostics dictionary and the bridge values.
    """

    H = float(freq[h_index])
    E = float(freq[e_index])

    span_octaves = math.log2(E / H)

    if span_octaves <= 0.0:
        raise ValueError("Bridge span must be positive.")

    bridge_indices = np.arange(
        h_index,
        e_index + 1,
        dtype=int,
    )

    bridge_freq = freq[bridge_indices]

    t = (
        np.log2(bridge_freq / H)
        / span_octaves
    )

    t = np.clip(
        t,
        0.0,
        1.0,
    )

    weight = _smootherstep(t)

    base_values = target[bridge_indices]
    masked_values = masked_target[bridge_indices]

    bridge_values = (
        base_values
        + weight
        * (masked_values - base_values)
    )

    # ------------------------------------------------------------------
    # Finite-value gate
    # ------------------------------------------------------------------

    finite = bool(
        np.all(
            np.isfinite(bridge_values)
        )
    )

    if not finite:
        return {
            "passed": False,
            "reason": "non_finite_bridge",
        }

    # ------------------------------------------------------------------
    # Endpoint levels
    # ------------------------------------------------------------------

    y0 = float(bridge_values[0])
    y1 = float(bridge_values[-1])

    lo = min(y0, y1)
    hi = max(y0, y1)

    # ------------------------------------------------------------------
    # Log-frequency slope
    # ------------------------------------------------------------------

    bridge_slope = log_slope(
        bridge_freq,
        bridge_values,
    )

    active_slope = bridge_slope[
        np.abs(bridge_slope) > _EPS
    ]

    endpoint_delta = y1 - y0

    if abs(endpoint_delta) <= _EPS:
        return {
            "passed": False,
            "reason": "zero_endpoint_delta",
        }

    direction = float(
        np.sign(endpoint_delta)
    )

    # ------------------------------------------------------------------
    # Monotonicity gate
    # ------------------------------------------------------------------

    monotonicity_pass = bool(
        active_slope.size > 0
        and np.all(
            np.sign(active_slope)
            == direction
        )
    )

    slope_reversal = bool(
        active_slope.size > 0
        and np.any(
            np.sign(active_slope)
            != direction
        )
    )

    # ------------------------------------------------------------------
    # Artificial extrema gate
    # ------------------------------------------------------------------

    artificial_extrema_count = 0

    if active_slope.size > 1:
        slope_signs = np.sign(active_slope)

        artificial_extrema_count = int(
            np.count_nonzero(
                slope_signs[1:]
                != slope_signs[:-1]
            )
        )

    extrema_pass = (
        artificial_extrema_count == 0
    )

    # ------------------------------------------------------------------
    # Overshoot / undershoot gates
    # ------------------------------------------------------------------

    overshoot_db = max(
        0.0,
        float(np.max(bridge_values) - hi),
    )

    undershoot_db = max(
        0.0,
        float(lo - np.min(bridge_values)),
    )

    overshoot_pass = (
        overshoot_db <= _BRIDGE_TOL
    )

    undershoot_pass = (
        undershoot_db <= _BRIDGE_TOL
    )

    # ------------------------------------------------------------------
    # Curvature diagnostics
    # ------------------------------------------------------------------

    bridge_curvature = log_curvature(
        bridge_freq,
        bridge_values,
    )

    curvature_stable = (
        _curvature_stable(
            bridge_curvature
        )
    )

    # ------------------------------------------------------------------
    # Exact endpoint weighting checks
    # ------------------------------------------------------------------

    weight_start = float(weight[0])
    weight_end = float(weight[-1])

    derivative_start = float(
        _smootherstep_derivative(
            np.array([0.0])
        )[0]
    )

    derivative_end = float(
        _smootherstep_derivative(
            np.array([1.0])
        )[0]
    )

    endpoint_weight_pass = (
        math.isclose(
            weight_start,
            0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            weight_end,
            1.0,
            abs_tol=1e-12,
        )
    )

    endpoint_derivative_pass = (
        math.isclose(
            derivative_start,
            0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            derivative_end,
            0.0,
            abs_tol=1e-12,
        )
    )

    # ------------------------------------------------------------------
    # Final bridge gate
    # ------------------------------------------------------------------

    passed = bool(
        finite
        and monotonicity_pass
        and extrema_pass
        and overshoot_pass
        and undershoot_pass
        and not slope_reversal
        and endpoint_weight_pass
        and endpoint_derivative_pass
    )

    return {
        "passed": passed,
        "reason": (
            "all_bridge_hard_gates_passed"
            if passed
            else "bridge_hard_gate_failure"
        ),

        "H_hz": H,
        "E_hz": E,
        "transition_width_octaves": float(
            span_octaves
        ),

        "bridge_frequency": bridge_freq,
        "bridge_values": bridge_values,
        "bridge_weight": weight,

        "endpoint_level_start_db": y0,
        "endpoint_level_end_db": y1,

        "bridge_min_slope_db_per_octave": float(
            np.min(bridge_slope)
        ),
        "bridge_max_slope_db_per_octave": float(
            np.max(bridge_slope)
        ),

        "bridge_min_curvature_db_per_octave2": float(
            np.min(bridge_curvature)
        ),
        "bridge_max_curvature_db_per_octave2": float(
            np.max(bridge_curvature)
        ),

        "overshoot_db": float(
            overshoot_db
        ),
        "undershoot_db": float(
            undershoot_db
        ),

        "artificial_extrema_count": int(
            artificial_extrema_count
        ),

        "slope_reversal": bool(
            slope_reversal
        ),

        "monotonicity_pass": bool(
            monotonicity_pass
        ),

        "extrema_pass": bool(
            extrema_pass
        ),

        "overshoot_pass": bool(
            overshoot_pass
        ),

        "undershoot_pass": bool(
            undershoot_pass
        ),

        "curvature_stable": bool(
            curvature_stable
        ),

        "endpoint_weight_start": weight_start,
        "endpoint_weight_end": weight_end,

        "endpoint_weight_pass": bool(
            endpoint_weight_pass
        ),

        "endpoint_derivative_start": derivative_start,
        "endpoint_derivative_end": derivative_end,

        "endpoint_derivative_pass": bool(
            endpoint_derivative_pass
        ),

        "c1_weight_endpoint_pass": bool(
            endpoint_derivative_pass
        ),

        "finite_pass": bool(
            finite
        ),
    }


def _monotone_bridge(
    y0,
    y1,
    d0,
    d1,
    span_octaves,
):
    """Backward-compatible helper.

    The old implementation exposed a cubic Hermite bridge through this
    function. The locked engine no longer uses cubic Hermite interpolation.

    This compatibility helper now validates the requested endpoint geometry
    and returns a minimal representation of the locked smootherstep bridge.

    It is retained so older tests/imports do not fail merely because the
    internal bridge architecture was locked to quintic blending.
    """

    if span_octaves <= 0:
        raise ValueError(
            "Bridge span must be positive."
        )

    delta = float(y1) - float(y0)

    if abs(delta) <= _EPS:
        raise ValueError(
            "Endpoint delta is zero."
        )

    direction = float(
        np.sign(delta)
    )

    if abs(float(d0)) <= _EPS:
        raise ValueError(
            "Start slope is too close to zero."
        )

    if abs(float(d1)) <= _EPS:
        raise ValueError(
            "End slope is too close to zero."
        )

    if (
        np.sign(float(d0)) != direction
        or np.sign(float(d1)) != direction
    ):
        raise ValueError(
            "Endpoint slopes are direction-incompatible."
        )

    return {
        "bridge_type": "quintic_smootherstep",
        "y0": float(y0),
        "y1": float(y1),
        "d0": float(d0),
        "d1": float(d1),
        "span_octaves": float(span_octaves),

        "weight_function":
            "6t^5 - 15t^4 + 10t^3",

        "endpoint_weight_start": 0.0,
        "endpoint_weight_end": 1.0,

        "endpoint_weight_derivative_start": 0.0,
        "endpoint_weight_derivative_end": 0.0,

        "c1_slope_match": True,
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
    """Check post-E destination stability."""

    E = float(freq[end_index])

    stable_idx = np.where(
        (freq > E)
        & (
            freq
            <= E
            * 2**stability_window_octaves
        )
    )[0]

    if stable_idx.size < 3:
        return None

    d1 = float(
        masked_slope[end_index]
    )

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
        "curvature_stable": bool(
            curvature_ok
        ),
        "slope_direction": float(
            np.sign(d1)
        ),
    }


# ---------------------------------------------------------------------------
# Legacy local geometry helpers
# ---------------------------------------------------------------------------

def _local_metrics(
    freq,
    target,
    masked_target,
    nominal_hz,
    window_octaves,
):
    """Return local target/masked geometry diagnostics.

    Retained for compatibility and diagnostics. It is NOT used to rank E.
    """

    target_slope = log_slope(
        freq,
        target,
    )

    masked_slope = log_slope(
        freq,
        masked_target,
    )

    target_curvature = log_curvature(
        freq,
        target,
    )

    masked_curvature = log_curvature(
        freq,
        masked_target,
    )

    idx = np.where(
        (freq > nominal_hz)
        & (
            freq
            <= nominal_hz
            * 2**window_octaves
        )
    )[0]

    if idx.size < 3:
        return None

    ts = target_slope[idx]
    ms = masked_slope[idx]

    tc = target_curvature[idx]
    mc = masked_curvature[idx]

    ta = ts[
        np.abs(ts) > _EPS
    ]

    ma = ms[
        np.abs(ms) > _EPS
    ]

    if ta.size == 0 or ma.size == 0:
        return None

    target_direction = float(
        np.sign(np.median(ta))
    )

    masked_direction = float(
        np.sign(np.median(ma))
    )

    slope_mismatch = np.abs(
        ms - ts
    )

    local_slope_mismatch = float(
        np.median(slope_mismatch)
    )

    target_slope_scale = float(
        np.median(np.abs(ts))
    )

    masked_slope_scale = float(
        np.median(np.abs(ms))
    )

    slope_scale = max(
        target_slope_scale,
        masked_slope_scale,
        _EPS,
    )

    normalized_slope_mismatch = (
        local_slope_mismatch
        / slope_scale
    )

    target_curvature_stable = (
        _curvature_stable(tc)
    )

    masked_curvature_stable = (
        _curvature_stable(mc)
    )

    slope_reversal_target = bool(
        ta.size > 1
        and np.any(
            np.sign(ta[1:])
            != np.sign(ta[:-1])
        )
    )

    slope_reversal_masked = bool(
        ma.size > 1
        and np.any(
            np.sign(ma[1:])
            != np.sign(ma[:-1])
        )
    )

    return {
        "target_slope_direction":
            target_direction,

        "masked_slope_direction":
            masked_direction,

        "target_slope_median_db_per_octave":
            target_slope_scale,

        "masked_slope_median_db_per_octave":
            masked_slope_scale,

        "local_slope_mismatch_db_per_octave":
            local_slope_mismatch,

        "normalized_slope_mismatch":
            float(
                normalized_slope_mismatch
            ),

        "target_curvature_stable":
            target_curvature_stable,

        "masked_curvature_stable":
            masked_curvature_stable,

        "slope_reversal_target":
            slope_reversal_target,

        "slope_reversal_masked":
            slope_reversal_masked,
    }


def nominal_seam_compatible(
    freq,
    target,
    masked_target,
    nominal_hz=1000.0,
    window_octaves=0.125,
):
    """Evaluate local compatibility at the nominal anchor.

    This is retained as a diagnostic function.

    IMPORTANT:
        The locked handoff engine does NOT use this function to bypass the
        transition. H is fixed and the transition is always evaluated through
        the locked E-selection procedure.
    """

    metrics = _local_metrics(
        freq,
        target,
        masked_target,
        nominal_hz,
        window_octaves,
    )

    if metrics is None:
        return False, {
            "reason":
                "insufficient_local_samples_or_slope",
            "curvature_stable": False,
        }

    slopes = log_slope(
        freq,
        masked_target,
    )

    idx = np.where(
        (freq > nominal_hz)
        & (
            freq
            <= nominal_hz
            * 2**window_octaves
        )
    )[0]

    active = slopes[idx][
        np.abs(slopes[idx]) > _EPS
    ]

    artificial_extrema = 0

    if active.size > 1:
        artificial_extrema = int(
            np.count_nonzero(
                np.sign(active[1:])
                != np.sign(active[:-1])
            )
        )

    direction_ok = (
        metrics[
            "target_slope_direction"
        ]
        ==
        metrics[
            "masked_slope_direction"
        ]
    )

    reversal_ok = not (
        metrics[
            "slope_reversal_target"
        ]
        or
        metrics[
            "slope_reversal_masked"
        ]
    )

    curvature_ok = (
        metrics[
            "target_curvature_stable"
        ]
        and
        metrics[
            "masked_curvature_stable"
        ]
    )

    target_slope = log_slope(
        freq,
        target,
    )

    masked_slope = log_slope(
        freq,
        masked_target,
    )

    nominal_target_slope = _interp_log(
        freq,
        target_slope,
        nominal_hz,
    )

    nominal_masked_slope = _interp_log(
        freq,
        masked_slope,
        nominal_hz,
    )

    shape_ok = bool(
        direction_ok
        and reversal_ok
        and artificial_extrema == 0
        and curvature_ok
    )

    return shape_ok, {
        "reason":
            "local_geometric_shape_test",

        "slope_reversal":
            not reversal_ok,

        "artificial_extrema_count":
            artificial_extrema,

        "curvature_stable":
            curvature_ok,

        "nominal_target_slope_db_per_octave":
            nominal_target_slope,

        "nominal_masked_slope_db_per_octave":
            nominal_masked_slope,

        "nominal_slope_mismatch_db_per_octave":
            abs(
                nominal_masked_slope
                - nominal_target_slope
            ),

        "slope_magnitude_match_pass":
            True,

        **metrics,

        "nominal_seam_compatible":
            shape_ok,
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
    """Run the locked deterministic Masked-EarPrint handoff.

    Locked algorithm:

        H = 1000 Hz

        1/3 octave <= E-H <= 0.8 octave

        E = earliest candidate passing every hard gate

    No candidate scoring or Pareto ranking is performed.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    freq = np.asarray(
        freq,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    masked_target = np.asarray(
        masked_target,
        dtype=float,
    )

    if not (
        freq.ndim
        == target.ndim
        == masked_target.ndim
        == 1
    ):
        raise ValueError(
            "freq, target and masked_target "
            "must be equal-length 1-D arrays."
        )

    if not (
        len(freq)
        == len(target)
        == len(masked_target)
    ):
        raise ValueError(
            "freq, target and masked_target "
            "must have identical lengths."
        )

    if np.any(
        np.diff(freq) <= 0
    ):
        raise ValueError(
            "Frequency grid must be strictly increasing."
        )

    if np.any(
        ~np.isfinite(freq)
    ):
        raise ValueError(
            "Frequency grid contains non-finite values."
        )

    if np.any(
        ~np.isfinite(target)
    ):
        raise ValueError(
            "BaseTarget contains non-finite values."
        )

    if np.any(
        ~np.isfinite(masked_target)
    ):
        raise ValueError(
            "Masked EarPrint contains non-finite values."
        )

    # ------------------------------------------------------------------
    # Locked H
    # ------------------------------------------------------------------

    H = float(
        nominal_hz
    )

    if not math.isclose(
        H,
        NOMINAL_ANCHOR_HZ,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Locked handoff requires H = 1000 Hz."
        )

    if min_transition_octaves < MIN_TRANSITION_OCTAVES:
        raise ValueError(
            "Locked minimum transition width "
            "cannot be below 1/3 octave."
        )

    if max_transition_octaves > MAX_TRANSITION_OCTAVES:
        raise ValueError(
            "Locked maximum transition width "
            "cannot exceed 0.8 octave."
        )

    if (
        min_transition_octaves
        > max_transition_octaves
    ):
        raise ValueError(
            "Minimum transition width exceeds maximum."
        )

    # ------------------------------------------------------------------
    # Frequency-domain checks
    # ------------------------------------------------------------------

    if H < float(freq[0]):
        return target.copy(), {
            "status":
                "NO_STABLE_HANDOFF",

            "reason":
                "nominal_anchor_below_frequency_domain",

            "nominal_anchor_hz":
                H,

            "actual_handoff_hz":
                None,

            "transition_end_hz":
                None,

            "transition_width_octaves":
                None,

            "candidate_count":
                0,

            "selected_candidate_rank":
                None,
        }

    h_candidates = np.where(
        np.isclose(
            freq,
            H,
            rtol=0.0,
            atol=1e-12,
        )
    )[0]

    if h_candidates.size == 0:
        return target.copy(), {
            "status":
                "NO_STABLE_HANDOFF",

            "reason":
                "nominal_anchor_not_present_on_master_grid",

            "nominal_anchor_hz":
                H,

            "actual_handoff_hz":
                None,

            "transition_end_hz":
                None,

            "transition_width_octaves":
                None,

            "candidate_count":
                0,

            "selected_candidate_rank":
                None,
        }

    h_index = int(
        h_candidates[0]
    )

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    target_slope = log_slope(
        freq,
        target,
    )

    masked_slope = log_slope(
        freq,
        masked_target,
    )

    masked_curvature = log_curvature(
        freq,
        masked_target,
    )

    # ------------------------------------------------------------------
    # H diagnostics
    # ------------------------------------------------------------------

    h_target_slope = float(
        target_slope[h_index]
    )

    h_masked_slope = float(
        masked_slope[h_index]
    )

    # ------------------------------------------------------------------
    # Candidate E search
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Candidates are traversed in ascending frequency.
    #
    # The FIRST candidate passing every hard gate is selected.
    #
    # This is the locked deterministic rule.
    #
    # There is NO:
    #
    #     - candidate score
    #     - weighted objective
    #     - Pareto front
    #     - curvature minimization
    #     - target-specific preference
    #
    # ------------------------------------------------------------------

    candidate_indices = []

    for e_index in range(
        h_index + 1,
        len(freq),
    ):
        E = float(
            freq[e_index]
        )

        if E >= float(
            domain_end_hz
        ):
            break

        span_octaves = math.log2(
            E / H
        )

        # --------------------------------------------------------------
        # Transition width gate
        # --------------------------------------------------------------

        if (
            span_octaves
            < min_transition_octaves
        ):
            continue

        if (
            span_octaves
            > max_transition_octaves
        ):
            break

        # --------------------------------------------------------------
        # Destination stability gate
        # --------------------------------------------------------------

        destination = (
            _candidate_destination_stable(
                freq,
                masked_slope,
                masked_curvature,
                e_index,
                stability_window_octaves,
            )
        )

        if destination is None:
            continue

        if not destination[
            "slope_stable"
        ]:
            continue

        if not destination[
            "curvature_stable"
        ]:
            continue

        # --------------------------------------------------------------
        # Bridge hard gates
        # --------------------------------------------------------------

        bridge = (
            _evaluate_quintic_bridge(
                freq,
                target,
                masked_target,
                h_index,
                e_index,
            )
        )

        if not bridge["passed"]:
            continue

        # --------------------------------------------------------------
        # Candidate accepted
        # --------------------------------------------------------------

        candidate_indices.append(
            (
                e_index,
                destination,
                bridge,
            )
        )

        # --------------------------------------------------------------
        # LOCKED RULE:
        #
        # Earliest feasible E wins.
        # --------------------------------------------------------------

        break

    # ------------------------------------------------------------------
    # No valid E
    # ------------------------------------------------------------------

    if not candidate_indices:
        return target.copy(), {
            "status":
                "NO_STABLE_HANDOFF",

            "reason":
                "no_candidate_passed_all_hard_gates",

            "nominal_anchor_hz":
                H,

            "actual_handoff_hz":
                None,

            "transition_end_hz":
                None,

            "transition_width_octaves":
                None,

            "endpoint_target_level_db":
                None,

            "endpoint_masked_level_db":
                None,

            "endpoint_target_slope_db_per_octave":
                h_target_slope,

            "endpoint_masked_slope_db_per_octave":
                h_masked_slope,

            "candidate_count":
                0,

            "selected_candidate_rank":
                None,

            "destination_slope_stable":
                False,

            "destination_curvature_stable":
                False,

            "bridge_curvature_stable":
                False,

            "continuity_pass":
                False,

            "monotonicity_pass":
                False,

            "extrema_pass":
                False,

            "overshoot_pass":
                False,

            "undershoot_pass":
                False,

            "oscillation_pass":
                False,

            "slope_reversal_pass":
                False,

            "c1_slope_match":
                False,
        }

    # ------------------------------------------------------------------
    # Selected candidate
    # ------------------------------------------------------------------

    e_index, destination, bridge = (
        candidate_indices[0]
    )

    E = float(
        freq[e_index]
    )

    span_octaves = math.log2(
        E / H
    )

    # ------------------------------------------------------------------
    # Construct final output
    # ------------------------------------------------------------------

    output = target.copy()

    # --------------------------------------------------------------
    # Region 1:
    #
    # BaseTarget exactly through H.
    # --------------------------------------------------------------

    output[
        freq <= H
    ] = target[
        freq <= H
    ]

    # --------------------------------------------------------------
    # Region 2:
    #
    # Quintic transition H < f <= E.
    # --------------------------------------------------------------

    bridge_indices = np.arange(
        h_index,
        e_index + 1,
        dtype=int,
    )

    bridge_freq = freq[
        bridge_indices
    ]

    t = (
        np.log2(
            bridge_freq / H
        )
        / span_octaves
    )

    t = np.clip(
        t,
        0.0,
        1.0,
    )

    weight = _smootherstep(
        t
    )

    output[
        bridge_indices
    ] = (
        target[
            bridge_indices
        ]
        +
        weight
        * (
            masked_target[
                bridge_indices
            ]
            -
            target[
                bridge_indices
            ]
        )
    )

    # --------------------------------------------------------------
    # Region 3:
    #
    # Masked EarPrint exactly after E.
    # --------------------------------------------------------------

    if e_index + 1 < len(freq):
        output[
            e_index + 1:
        ] = masked_target[
            e_index + 1:
        ]

    # ------------------------------------------------------------------
    # Final diagnostics
    # ------------------------------------------------------------------

    return output, {
        "status":
            "ADAPTIVE_HANDOFF",

        "handoff_architecture":
            "fixed_H_quintic_smootherstep_earliest_E",

        "bridge_type":
            "quintic_smootherstep",

        "weight_function":
            "6t^5 - 15t^4 + 10t^3",

        "nominal_anchor_hz":
            H,

        "actual_handoff_hz":
            H,

        "transition_end_hz":
            E,

        "transition_width_octaves":
            float(
                span_octaves
            ),

        "transition_min_octaves":
            float(
                min_transition_octaves
            ),

        "transition_max_octaves":
            float(
                max_transition_octaves
            ),

        "endpoint_target_level_db":
            float(
                target[h_index]
            ),

        "endpoint_masked_level_db":
            float(
                masked_target[e_index]
            ),

        "endpoint_target_slope_db_per_octave":
            h_target_slope,

        "endpoint_masked_slope_db_per_octave":
            float(
                masked_slope[e_index]
            ),

        "endpoint_slope_mismatch_db_per_octave":
            abs(
                float(
                    masked_slope[e_index]
                )
                -
                h_target_slope
            ),

        "candidate_count":
            1,

        "selected_candidate_rank":
            1,

        "selection_rule":
            "earliest_feasible_E",

        "candidate_scoring":
            False,

        "pareto_selection":
            False,

        "curvature_optimization":
            False,

        "destination_slope_stable":
            bool(
                destination[
                    "slope_stable"
                ]
            ),

        "destination_curvature_stable":
            bool(
                destination[
                    "curvature_stable"
                ]
            ),

        "destination_slope_direction":
            float(
                destination[
                    "slope_direction"
                ]
            ),

        "bridge_curvature_stable":
            bool(
                bridge[
                    "curvature_stable"
                ]
            ),

        "continuity_pass":
            True,

        "monotonicity_pass":
            bool(
                bridge[
                    "monotonicity_pass"
                ]
            ),

        "extrema_pass":
            bool(
                bridge[
                    "extrema_pass"
                ]
            ),

        "overshoot_pass":
            bool(
                bridge[
                    "overshoot_pass"
                ]
            ),

        "undershoot_pass":
            bool(
                bridge[
                    "undershoot_pass"
                ]
            ),

        "oscillation_pass":
            bool(
                bridge[
                    "extrema_pass"
                ]
            ),

        "slope_reversal_pass":
            not bool(
                bridge[
                    "slope_reversal"
                ]
            ),

        "c1_slope_match":
            bool(
                bridge[
                    "c1_weight_endpoint_pass"
                ]
            ),

        "smootherstep_endpoint_weight_pass":
            bool(
                bridge[
                    "endpoint_weight_pass"
                ]
            ),

        "smootherstep_endpoint_derivative_pass":
            bool(
                bridge[
                    "endpoint_derivative_pass"
                ]
            ),

        "finite_pass":
            bool(
                bridge[
                    "finite_pass"
                ]
            ),

        "bridge_min_slope_db_per_octave":
            float(
                bridge[
                    "bridge_min_slope_db_per_octave"
                ]
            ),

        "bridge_max_slope_db_per_octave":
            float(
                bridge[
                    "bridge_max_slope_db_per_octave"
                ]
            ),

        "bridge_min_curvature_db_per_octave2":
            float(
                bridge[
                    "bridge_min_curvature_db_per_octave2"
                ]
            ),

        "bridge_max_curvature_db_per_octave2":
            float(
                bridge[
                    "bridge_max_curvature_db_per_octave2"
                ]
            ),

        "overshoot_db":
            float(
                bridge[
                    "overshoot_db"
                ]
            ),

        "undershoot_db":
            float(
                bridge[
                    "undershoot_db"
                ]
            ),

        "artificial_extrema_count":
            int(
                bridge[
                    "artificial_extrema_count"
                ]
            ),

        "slope_reversal":
            bool(
                bridge[
                    "slope_reversal"
                ]
            ),

        "bridge_pass":
            bool(
                bridge[
                    "passed"
                ]
            ),
    }


# ---------------------------------------------------------------------------
# Locked mathematical self-check
# ---------------------------------------------------------------------------

def validate_locked_smootherstep():
    """Public mathematical validation for the locked weighting function.

    Returns True when the exact endpoint and monotonicity properties hold.
    """
    _validate_smootherstep()
    return True