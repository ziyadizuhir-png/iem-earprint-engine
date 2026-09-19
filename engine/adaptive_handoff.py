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
