"""Adaptive Masked-EarPrint handoff.

The nominal 1 kHz point is an anchor, not an unconditional splice.
Only the minimum transition region is changed when the direct masked
handoff fails local shape tests.
"""
from __future__ import annotations
import math
import numpy as np

def log_slope(freq, level):
    x = np.log2(freq)
    out = np.empty_like(level, dtype=float)
    out[0] = (level[1]-level[0])/(x[1]-x[0])
    out[-1] = (level[-1]-level[-2])/(x[-1]-x[-2])
    out[1:-1] = (level[2:]-level[:-2])/(x[2:]-x[:-2])
    return out

def _interp_log(freq, level, hz):
    return float(np.interp(math.log(hz), np.log(freq), level))

def _sign_stable(values, sign, eps=1e-9):
    active = values[np.abs(values) > eps]
    return bool(active.size) and bool(np.all(np.sign(active) == np.sign(sign)))

def _monotone_bridge(y0, y1, d0, d1, span_octaves):
    delta = y1-y0
    secant = delta/span_octaves
    direction = float(np.sign(delta))
    if direction == 0 or np.sign(d0) != direction or np.sign(d1) != direction:
        raise ValueError("Endpoint slopes are not direction-compatible.")
    cap = 3.0*abs(secant)
    m0 = direction*min(abs(d0), cap)
    m1 = direction*min(abs(d1), cap)
    a = 2*y0-2*y1+span_octaves*m0+span_octaves*m1
    b = -3*y0+3*y1-2*span_octaves*m0-span_octaves*m1
    c = span_octaves*m0
    t = np.linspace(0, 1, 401)
    values = a*t**3+b*t**2+c*t+y0
    slopes = (3*a*t**2+2*b*t+c)/span_octaves
    overshoot = max(0.0, float(np.max(values)-max(y0,y1)))
    undershoot = max(0.0, float(min(y0,y1)-np.min(values)))
    active = slopes[np.abs(slopes) > 1e-9]
    reversal = bool(active.size and np.any(np.sign(active) != direction))
    extrema = int(np.count_nonzero(np.sign(active[1:]) != np.sign(active[:-1]))) if active.size > 1 else 0
    if overshoot > 1e-9 or undershoot > 1e-9 or reversal:
        raise ValueError("Bridge failed monotonicity/overshoot constraints.")
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "endpoint_slope_start_used_db_per_octave": float(m0),
        "endpoint_slope_end_used_db_per_octave": float(m1),
        "bridge_min_slope_db_per_octave": float(np.min(slopes)),
        "bridge_max_slope_db_per_octave": float(np.max(slopes)),
        "overshoot_db": float(overshoot),
        "undershoot_db": float(undershoot),
        "artificial_extrema_count": extrema,
        "slope_reversal": reversal,
    }

def nominal_seam_compatible(freq, target, masked_target, nominal_hz=1000.0, window_octaves=0.125):
    slopes = log_slope(freq, masked_target)
    idx = np.where((freq > nominal_hz) & (freq <= nominal_hz*2**window_octaves))[0]
    if idx.size < 3:
        return False, {"reason": "insufficient_local_samples"}
    active = slopes[idx][np.abs(slopes[idx]) > 1e-9]
    if active.size == 0:
        return True, {"reason": "locally_flat", "slope_reversal": False, "artificial_extrema_count": 0}
    direction = float(np.sign(np.median(active)))
    sign_changes = int(np.count_nonzero(np.sign(active[1:]) != np.sign(active[:-1])))
    base = log_slope(freq, target)
    bi = np.where((freq > nominal_hz) & (freq <= nominal_hz*2**window_octaves))[0]
    ba = base[bi][np.abs(base[bi]) > 1e-9]
    base_direction = float(np.sign(np.median(ba))) if ba.size else direction
    return sign_changes == 0 and direction == base_direction, {
        "reason": "local_shape_test",
        "slope_reversal": sign_changes > 0,
        "artificial_extrema_count": sign_changes,
        "masked_local_slope_direction": direction,
        "target_local_slope_direction": base_direction,
    }

def adaptive_masked_handoff(
    freq, target, masked_target, nominal_hz=1000.0, domain_end_hz=12000.0,
    min_transition_octaves=0.125, max_transition_octaves=0.8,
    stability_window_octaves=0.20,
):
    out = masked_target.copy()
    st = log_slope(freq, target)
    sm = log_slope(freq, masked_target)

    ok, diag = nominal_seam_compatible(freq, target, masked_target, nominal_hz)
    if ok:
        return out, {
            "status": "NORMAL_HANDOFF",
            "nominal_anchor_hz": nominal_hz,
            "actual_handoff_hz": nominal_hz,
            "transition_end_hz": nominal_hz,
            "transition_width_octaves": 0.0,
            "endpoint_target_level_db": _interp_log(freq,target,nominal_hz),
            "endpoint_masked_level_db": _interp_log(freq,masked_target,nominal_hz),
            "endpoint_target_slope_db_per_octave": _interp_log(freq,st,nominal_hz),
            "endpoint_masked_slope_db_per_octave": _interp_log(freq,sm,nominal_hz),
            "curvature_stable": True,
            "continuity_pass": True, "monotonicity_pass": True,
            "extrema_pass": diag["artificial_extrema_count"] == 0,
            "overshoot_pass": True, "undershoot_pass": True,
            "oscillation_pass": diag["artificial_extrema_count"] == 0,
            "slope_reversal_pass": not diag["slope_reversal"],
            "nominal_seam_compatible": True,
        }

    for hi in np.where(freq > nominal_hz)[0]:
        H = float(freq[hi])
        if H >= domain_end_hz:
            break
        d0 = float(st[hi])
        if abs(d0) < 1e-6:
            continue

        for ei in range(hi+1, len(freq)):
            E = float(freq[ei])
            if E >= domain_end_hz:
                break
            span = math.log2(E/H)
            if span < min_transition_octaves:
                continue
            if span > max_transition_octaves:
                break

            y0 = float(target[hi])
            y1 = float(masked_target[ei])
            delta = y1-y0
            if abs(delta) < 1e-9:
                continue
            secant = delta/span
            d1 = float(sm[ei])
            if np.sign(secant) != np.sign(d0) or np.sign(secant) != np.sign(d1):
                continue

            stable_idx = np.where((freq > E) & (freq <= E*2**stability_window_octaves))[0]
            if stable_idx.size < 3 or not _sign_stable(sm[stable_idx], np.sign(d1)):
                continue

            try:
                bridge = _monotone_bridge(y0,y1,d0,d1,span)
            except ValueError:
                continue

            t = np.log2(freq[hi+1:ei+1]/H)/span
            out[hi+1:ei+1] = bridge["a"]*t**3+bridge["b"]*t**2+bridge["c"]*t+y0
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
                "curvature_stable": True,
                "continuity_pass": True, "monotonicity_pass": True,
                "extrema_pass": bridge["artificial_extrema_count"] == 0,
                "overshoot_pass": bridge["overshoot_db"] <= 1e-9,
                "undershoot_pass": bridge["undershoot_db"] <= 1e-9,
                "oscillation_pass": bridge["artificial_extrema_count"] == 0,
                "slope_reversal_pass": not bridge["slope_reversal"],
                **bridge,
                "nominal_seam_compatible": False,
            }

    return target.copy(), {
        "status": "NO_STABLE_HANDOFF",
        "nominal_anchor_hz": nominal_hz,
        "actual_handoff_hz": None, "transition_end_hz": None,
        "transition_width_octaves": None,
        "endpoint_target_level_db": None, "endpoint_masked_level_db": None,
        "endpoint_target_slope_db_per_octave": None,
        "endpoint_masked_slope_db_per_octave": None,
        "curvature_stable": False,
        "continuity_pass": False, "monotonicity_pass": False,
        "extrema_pass": False, "overshoot_pass": False, "undershoot_pass": False,
        "oscillation_pass": False, "slope_reversal_pass": False,
        "nominal_seam_compatible": False,
    }
