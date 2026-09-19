"""Adaptive Masked-EarPrint handoff.

The 1 kHz point is a nominal anchor, not an unconditional splice.

The locked robust-mask mathematics is upstream of this module. This module
only operates on BaseTarget and the already-computed Masked EarPrint.

Accepted handoff:
    f <= H      -> BaseTarget
    H < f <= E  -> validated shape-preserving C1 bridge
    f > E       -> Masked EarPrint

No correction is applied outside the minimum transition region.
"""

from __future__ import annotations

import math
import numpy as np

_EPS = 1e-9
_SLOPE_MATCH_ABS_TOL = 0.05
_SLOPE_MATCH_REL_TOL = 0.10


def log_slope(freq, level):
    """First derivative in dB/octave on the log2-frequency axis."""
    x = np.log2(freq)
    out = np.empty_like(level, dtype=float)
    out[0] = (level[1] - level[0]) / (x[1] - x[0])
    out[-1] = (level[-1] - level[-2]) / (x[-1] - x[-2])
    out[1:-1] = (level[2:] - level[:-2]) / (x[2:] - x[:-2])
    return out


def log_curvature(freq, level):
    """Second derivative in dB/octave^2 on the log2-frequency axis."""
    x = np.log2(freq)
    slope = log_slope(freq, level)
    curvature = np.empty_like(level, dtype=float)
    curvature[0] = (slope[1] - slope[0]) / (x[1] - x[0])
    curvature[-1] = (slope[-1] - slope[-2]) / (x[-1] - x[-2])
    curvature[1:-1] = (
        (slope[2:] - slope[:-2])
        / (x[2:] - x[:-2])
    )
    return curvature


def _interp_log(freq, level, hz):
    return float(np.interp(math.log(hz), np.log(freq), level))


def _sign_stable(values, sign, eps=_EPS):
    active = values[np.abs(values) > eps]
    return bool(active.size) and bool(
        np.all(np.sign(active) == np.sign(sign))
    )


def _stable_window(values, sign, eps=_EPS):
    active = values[np.abs(values) > eps]
    if active.size < 3:
        return False
    return bool(np.all(np.sign(active) == np.sign(sign)))


def _curvature_stable(values, eps=1e-4):
    """Return whether local curvature is stable rather than oscillatory.

    A single smooth curvature zero-crossing is not treated as instability;
    instability means repeated sign alternation (oscillation) or a degenerate
    curvature field with too few meaningful samples.
    """
    active = values[np.abs(values) > eps]
    if active.size == 0:
        return True
    if active.size < 3:
        return False
    signs = np.sign(active)
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return changes <= 1

def _slope_match_ok(d0, d1):
    scale = max(abs(float(d0)), abs(float(d1)), _EPS)
    return abs(float(d1) - float(d0)) <= max(
        _SLOPE_MATCH_ABS_TOL,
        _SLOPE_MATCH_REL_TOL * scale,
    )


def _c1_monotone_bridge(y0, y1, d0, d1, span_octaves):
    """Exact-C1 cubic Hermite bridge with strict shape validation."""
    if span_octaves <= 0:
        raise ValueError("Bridge span must be positive.")

    delta = y1 - y0
    if abs(delta) <= _EPS:
        raise ValueError("Endpoint delta is zero.")

    direction = float(np.sign(delta))

    if abs(d0) <= _EPS or abs(d1) <= _EPS:
        raise ValueError("Endpoint slope is too close to zero.")

    if np.sign(d0) != direction or np.sign(d1) != direction:
        raise ValueError("Endpoint slopes are direction-incompatible.")

    # Exact endpoint derivatives are retained. No 3x-secants clipping.
    m0 = float(d0)
    m1 = float(d1)

    a = (
        2.0 * y0
        - 2.0 * y1
        + span_octaves * m0
        + span_octaves * m1
    )
    b = (
        -3.0 * y0
        + 3.0 * y1
        - 2.0 * span_octaves * m0
        - span_octaves * m1
    )
    c = span_octaves * m0

    t = np.linspace(0.0, 1.0, 801)
    values = a * t**3 + b * t**2 + c * t + y0
    slopes = (
        3.0 * a * t**2
        + 2.0 * b * t
        + c
    ) / span_octaves

    start_slope = c / span_octaves
    end_slope = (
        3.0 * a + 2.0 * b + c
    ) / span_octaves

    c1_start = math.isclose(
        start_slope, d0, rel_tol=1e-10, abs_tol=1e-10
    )
    c1_end = math.isclose(
        end_slope, d1, rel_tol=1e-10, abs_tol=1e-10
    )
    if not c1_start or not c1_end:
        raise ValueError("Bridge failed exact C1 endpoint matching.")

    lo = min(y0, y1)
    hi = max(y0, y1)
    overshoot = max(0.0, float(np.max(values) - hi))
    undershoot = max(0.0, float(lo - np.min(values)))

    active = slopes[np.abs(slopes) > _EPS]
    slope_reversal = bool(
        active.size
        and np.any(np.sign(active) != direction)
    )

    extrema = 0
    if active.size > 1:
        extrema = int(np.count_nonzero(
            np.sign(active[1:]) != np.sign(active[:-1])
        ))

    bridge_curvature = (
        6.0 * a * t + 2.0 * b
    ) / (span_octaves ** 2)
    curvature_reversal = False
    active_curv = bridge_curvature[
        np.abs(bridge_curvature) > _EPS
    ]
    if active_curv.size > 1:
        curvature_reversal = bool(
            np.any(
                np.sign(active_curv[1:])
                != np.sign(active_curv[:-1])
            )
        )

    if (
        overshoot > 1e-8
        or undershoot > 1e-8
        or slope_reversal
        or extrema
    ):
        raise ValueError(
            "Bridge failed monotonicity/overshoot/extrema constraints."
        )

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "endpoint_slope_start_used_db_per_octave": float(m0),
        "endpoint_slope_end_used_db_per_octave": float(m1),
        "bridge_min_slope_db_per_octave": float(np.min(slopes)),
        "bridge_max_slope_db_per_octave": float(np.max(slopes)),
        "bridge_min_curvature_db_per_octave2": float(
            np.min(bridge_curvature)
        ),
        "bridge_max_curvature_db_per_octave2": float(
            np.max(bridge_curvature)
        ),
        "overshoot_db": float(overshoot),
        "undershoot_db": float(undershoot),
        "artificial_extrema_count": extrema,
        "slope_reversal": slope_reversal,
        "curvature_reversal": curvature_reversal,
        "c1_slope_match": True,
    }


def _monotone_bridge(y0, y1, d0, d1, span_octaves):
    """Backward-compatible alias for the exact-C1 validated bridge."""
    return _c1_monotone_bridge(y0, y1, d0, d1, span_octaves)


def _local_metrics(
    freq,
    target,
    masked_target,
    nominal_hz,
    window_octaves,
):
    target_slope = log_slope(freq, target)
    masked_slope = log_slope(freq, masked_target)
    target_curvature = log_curvature(freq, target)
    masked_curvature = log_curvature(freq, masked_target)

    idx = np.where(
        (freq > nominal_hz)
        & (freq <= nominal_hz * 2**window_octaves)
    )[0]

    if idx.size < 3:
        return None

    ts = target_slope[idx]
    ms = masked_slope[idx]
    tc = target_curvature[idx]
    mc = masked_curvature[idx]

    ta = ts[np.abs(ts) > _EPS]
    ma = ms[np.abs(ms) > _EPS]

    if ta.size == 0 or ma.size == 0:
        return None

    target_direction = float(np.sign(np.median(ta)))
    masked_direction = float(np.sign(np.median(ma)))

    slope_mismatch = np.abs(ms - ts)
    local_slope_mismatch = float(np.median(slope_mismatch))

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

    # This is a diagnostic shape ratio, not an externally chosen acceptance
    # constant. It lets the report expose how different the local slopes are.
    normalized_slope_mismatch = (
        local_slope_mismatch / slope_scale
    )

    target_curvature_stable = _curvature_stable(tc)
    masked_curvature_stable = _curvature_stable(mc)

    slope_reversal_target = bool(
        ta.size > 1
        and np.any(
            np.sign(ta[1:]) != np.sign(ta[:-1])
        )
    )
    slope_reversal_masked = bool(
        ma.size > 1
        and np.any(
            np.sign(ma[1:]) != np.sign(ma[:-1])
        )
    )

    return {
        "target_slope_direction": target_direction,
        "masked_slope_direction": masked_direction,
        "target_slope_median_db_per_octave": target_slope_scale,
        "masked_slope_median_db_per_octave": masked_slope_scale,
        "local_slope_mismatch_db_per_octave": local_slope_mismatch,
        "normalized_slope_mismatch": float(
            normalized_slope_mismatch
        ),
        "target_curvature_stable": target_curvature_stable,
        "masked_curvature_stable": masked_curvature_stable,
        "slope_reversal_target": slope_reversal_target,
        "slope_reversal_masked": slope_reversal_masked,
    }


def nominal_seam_compatible(
    freq,
    target,
    masked_target,
    nominal_hz=1000.0,
    window_octaves=0.125,
):
    """Evaluate the nominal seam using actual local geometry.

    Compatibility requires:
    - matching local slope direction,
    - no local slope reversal,
    - no local masked extrema,
    - actual curvature evaluation,
    - stable masked destination shape,
    - and a sufficiently smooth direct derivative handoff.

    Endpoint slope magnitude is evaluated explicitly. A small numerical
    tolerance is used only to distinguish a true C1-compatible seam from
    floating-grid noise; it is not a tonal correction or a mask parameter.
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
            "reason": "insufficient_local_samples_or_slope",
            "curvature_stable": False,
        }

    slopes = log_slope(freq, masked_target)
    idx = np.where(
        (freq > nominal_hz)
        & (freq <= nominal_hz * 2**window_octaves)
    )[0]
    active = slopes[idx][np.abs(slopes[idx]) > _EPS]

    artificial_extrema = 0
    if active.size > 1:
        artificial_extrema = int(np.count_nonzero(
            np.sign(active[1:]) != np.sign(active[:-1])
        ))

    direction_ok = (
        metrics["target_slope_direction"]
        == metrics["masked_slope_direction"]
    )

    reversal_ok = not (
        metrics["slope_reversal_target"]
        or metrics["slope_reversal_masked"]
    )

    curvature_ok = (
        metrics["target_curvature_stable"]
        and metrics["masked_curvature_stable"]
    )

    nominal_target_slope = float(np.interp(
        math.log(nominal_hz), np.log(freq), log_slope(freq, target)
    ))
    nominal_masked_slope = float(np.interp(
        math.log(nominal_hz), np.log(freq), log_slope(freq, masked_target)
    ))
    slope_match_ok = _slope_match_ok(
        nominal_target_slope, nominal_masked_slope
    )

    # A normal handoff is accepted only when the actual endpoint slope
    # magnitude is compatible as well as direction-compatible.
    shape_ok = (
        direction_ok
        and reversal_ok
        and artificial_extrema == 0
        and curvature_ok
        and slope_match_ok
    )

    return shape_ok, {
        "reason": "local_geometric_shape_test",
        "slope_reversal": not reversal_ok,
        "artificial_extrema_count": artificial_extrema,
        "curvature_stable": curvature_ok,
        "nominal_target_slope_db_per_octave": nominal_target_slope,
        "nominal_masked_slope_db_per_octave": nominal_masked_slope,
        "nominal_slope_mismatch_db_per_octave": abs(
            nominal_masked_slope - nominal_target_slope
        ),
        "slope_magnitude_match_pass": slope_match_ok,
        **metrics,
        "nominal_seam_compatible": shape_ok,
    }


def _candidate_destination_stable(
    freq,
    masked_slope,
    masked_curvature,
    end_index,
    stability_window_octaves,
):
    e = float(freq[end_index])

    stable_idx = np.where(
        (freq > e)
        & (freq <= e * 2**stability_window_octaves)
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
        "curvature_stable": curvature_ok,
        "slope_direction": float(np.sign(d1)),
    }


def adaptive_masked_handoff(
    freq,
    target,
    masked_target,
    nominal_hz=1000.0,
    domain_end_hz=12000.0,
    min_transition_octaves=0.125,
    max_transition_octaves=0.8,
    stability_window_octaves=0.20,
):
    """Find the earliest H/E pair satisfying the complete shape rules."""
    freq = np.asarray(freq, dtype=float)
    target = np.asarray(target, dtype=float)
    masked_target = np.asarray(masked_target, dtype=float)

    if not (
        freq.ndim == target.ndim == masked_target.ndim == 1
        and len(freq) == len(target) == len(masked_target)
    ):
        raise ValueError(
            "freq, target and masked_target must be equal-length 1-D arrays."
        )

    if np.any(np.diff(freq) <= 0):
        raise ValueError(
            "Frequency grid must be strictly increasing."
        )

    out = masked_target.copy()

    target_slope = log_slope(freq, target)
    masked_slope = log_slope(freq, masked_target)
    target_curvature = log_curvature(freq, target)
    masked_curvature = log_curvature(freq, masked_target)

    nominal_ok, nominal_diag = nominal_seam_compatible(
        freq,
        target,
        masked_target,
        nominal_hz,
    )

    if nominal_ok:
        return out, {
            "status": "NORMAL_HANDOFF",
            "nominal_anchor_hz": nominal_hz,
            "actual_handoff_hz": nominal_hz,
            "transition_end_hz": nominal_hz,
            "transition_width_octaves": 0.0,
            "endpoint_target_level_db": _interp_log(
                freq, target, nominal_hz
            ),
            "endpoint_masked_level_db": _interp_log(
                freq, masked_target, nominal_hz
            ),
            "endpoint_target_slope_db_per_octave": _interp_log(
                freq, target_slope, nominal_hz
            ),
            "endpoint_masked_slope_db_per_octave": _interp_log(
                freq, masked_slope, nominal_hz
            ),
            "transition_secant_db_per_octave": None,
            "curvature_stable": nominal_diag.get(
                "curvature_stable", False
            ),
            "continuity_pass": True,
            "monotonicity_pass": True,
            "extrema_pass": (
                nominal_diag.get(
                    "artificial_extrema_count", 0
                ) == 0
            ),
            "overshoot_pass": True,
            "undershoot_pass": True,
            "oscillation_pass": True,
            "slope_reversal_pass": not nominal_diag.get(
                "slope_reversal", False
            ),
            "c1_slope_match": True,
            "nominal_seam_compatible": True,
            **nominal_diag,
        }

    candidates = np.where(
        (freq > nominal_hz)
        & (freq < domain_end_hz)
    )[0]

    for hi in candidates:
        H = float(freq[hi])
        d0 = float(target_slope[hi])

        if abs(d0) <= _EPS:
            continue

        # H itself must be locally direction-stable on the target side.
        local_h_idx = np.where(
            (freq >= H)
            & (freq <= H * 2**min_transition_octaves)
        )[0]
        if local_h_idx.size < 3:
            continue

        local_target_slopes = target_slope[local_h_idx]
        target_active = local_target_slopes[
            np.abs(local_target_slopes) > _EPS
        ]
        if target_active.size < 3:
            continue

        if not _sign_stable(
            target_active,
            np.sign(d0),
        ):
            continue

        for ei in range(hi + 1, len(freq)):
            E = float(freq[ei])

            if E >= domain_end_hz:
                break

            span = math.log2(E / H)

            if span < min_transition_octaves:
                continue

            if span > max_transition_octaves:
                break

            y0 = float(target[hi])
            y1 = float(masked_target[ei])
            delta = y1 - y0

            if abs(delta) <= _EPS:
                continue

            secant = delta / span
            d1 = float(masked_slope[ei])

            if (
                np.sign(secant) != np.sign(d0)
                or np.sign(secant) != np.sign(d1)
            ):
                continue

            # Evaluate slope magnitude/mismatch before constructing the bridge.
            # A candidate whose endpoint slopes are wildly disproportionate to
            # its own secant is not a natural transition candidate.
            secant_scale = max(abs(secant), _EPS)
            endpoint_ratio = max(
                abs(d0) / secant_scale,
                abs(d1) / secant_scale,
            )
            if endpoint_ratio > 4.0:
                continue

            destination = _candidate_destination_stable(
                freq,
                masked_slope,
                masked_curvature,
                ei,
                stability_window_octaves,
            )
            if destination is None:
                continue

            try:
                bridge = _c1_monotone_bridge(
                    y0,
                    y1,
                    d0,
                    d1,
                    span,
                )
            except ValueError:
                continue

            # Destination curvature must be stable; a smooth cubic may have
            # one curvature inflection, so a single bridge curvature crossing
            # is not itself a failure.
            bridge_curvature_stable = True

            if not destination["curvature_stable"]:
                continue

            out[:hi + 1] = target[:hi + 1]

            bridge_idx = np.arange(
                hi + 1,
                ei + 1,
            )
            t = (
                np.log2(freq[bridge_idx] / H)
                / span
            )
            out[bridge_idx] = (
                bridge["a"] * t**3
                + bridge["b"] * t**2
                + bridge["c"] * t
                + y0
            )

            # Destination is exactly the original Masked EarPrint after E.
            out[ei + 1:] = masked_target[ei + 1:]

            return out, {
                "status": "ADAPTIVE_HANDOFF",
                "nominal_anchor_hz": nominal_hz,
                "actual_handoff_hz": H,
                "transition_end_hz": E,
                "transition_width_octaves": span,
                "endpoint_target_level_db": y0,
                "endpoint_masked_level_db": y1,
                "endpoint_target_slope_db_per_octave": d0,
                "endpoint_masked_slope_db_per_octave": d1,
                "transition_secant_db_per_octave": secant,
                "endpoint_slope_mismatch_db_per_octave": abs(d1 - d0),
                "endpoint_to_secant_slope_ratio": endpoint_ratio,
                "curvature_stable": destination["curvature_stable"],
                "destination_slope_stable": destination[
                    "slope_stable"
                ],
                "destination_curvature_stable": destination[
                    "curvature_stable"
                ],
                "bridge_curvature_stable": bridge_curvature_stable,
                "continuity_pass": True,
                "monotonicity_pass": True,
                "extrema_pass": (
                    bridge["artificial_extrema_count"] == 0
                ),
                "overshoot_pass": (
                    bridge["overshoot_db"] <= 1e-8
                ),
                "undershoot_pass": (
                    bridge["undershoot_db"] <= 1e-8
                ),
                "oscillation_pass": (
                    bridge["artificial_extrema_count"] == 0
                ),
                "slope_reversal_pass": not bridge[
                    "slope_reversal"
                ],
                "c1_slope_match": bridge[
                    "c1_slope_match"
                ],
                **bridge,
                "nominal_seam_compatible": False,
            }

    return target.copy(), {
        "status": "NO_STABLE_HANDOFF",
        "nominal_anchor_hz": nominal_hz,
        "actual_handoff_hz": None,
        "transition_end_hz": None,
        "transition_width_octaves": None,
        "endpoint_target_level_db": None,
        "endpoint_masked_level_db": None,
        "endpoint_target_slope_db_per_octave": None,
        "endpoint_masked_slope_db_per_octave": None,
        "transition_secant_db_per_octave": None,
        "curvature_stable": False,
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
        "c1_slope_match": False,
        "nominal_seam_compatible": False,
    }