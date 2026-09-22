"""Acceptance tests for the locked TRUE-C1 adaptive handoff."""

from __future__ import annotations

import math
import numpy as np
import pytest

from engine.adaptive_handoff import (
    adaptive_masked_handoff,
    log_slope,
    _evaluate_exact_c1_bridge,
    _evaluate_monotone_cubic_bridge,
)


def grid(start=20.0, end=12000.0, ppo=192):
    n=int(round(math.log2(end/start)*ppo))+1
    return np.sort(np.unique(np.r_[np.geomspace(start,end,n),1000.0]))


def base(freq):
    x=np.log2(freq/1000.0)
    return 86.0 + 2.2*x + 0.05*x*x


def masked(freq, end=1700.0, amp=2.0):
    x=np.log2(freq/1000.0); span=math.log2(end/1000.0)
    t=np.clip(x/span,0,1)
    return base(freq)+amp*(3*t*t-2*t*t*t)


def test_locked_constants_and_bounds():
    freq=grid(); target=base(freq); m=masked(freq)
    out,d=adaptive_masked_handoff(freq,target,m)
    assert d["nominal_anchor_hz"]==1000.0
    assert d["selection_rule"]=="earliest_feasible_E"
    assert d["candidate_scoring"] is False
    assert d["pareto_selection"] is False
    assert d["curvature_optimization"] is False
    if d["status"]=="ADAPTIVE_HANDOFF":
        assert 1/3-1e-12 <= d["transition_width_octaves"] <= .8+1e-12


def test_base_preserved_through_H_and_mask_after_E():
    freq=grid(); target=base(freq); m=masked(freq)
    out,d=adaptive_masked_handoff(freq,target,m)
    assert np.array_equal(out[freq<=1000],target[freq<=1000])
    if d["status"]=="ADAPTIVE_HANDOFF":
        E=d["transition_end_hz"]
        assert np.array_equal(out[freq>E],m[freq>E])


def test_exact_c1_endpoint_slopes_are_retained():
    r=_evaluate_exact_c1_bridge(80.0,82.0,2.2,7.4,math.log2(1261.6488943/1000.0))
    assert r["c1_slope_match"]
    assert r["analytic_derivative_pass"]
    assert r["alpha"]>=0 and r["beta"]>=0
    assert r["alpha_plus_beta"]<=3.0+1e-12
    assert r["endpoint_slope_start_used_db_per_octave"]==2.2
    assert r["endpoint_slope_end_used_db_per_octave"]==7.4


def test_c1_rejects_when_normalized_endpoint_sum_exceeds_three():
    with pytest.raises(ValueError):
        _evaluate_exact_c1_bridge(80.0,82.0,2.2,16.0,math.log2(1261.6488943/1000.0))


def test_analytic_derivative_gate_is_used():
    freq=np.geomspace(1000,1700,97); target=80+2*np.log2(freq/1000); m=target+1.5*(3*(np.log2(freq/1000)/math.log2(1.7))**2-2*(np.log2(freq/1000)/math.log2(1.7))**3)
    d=_evaluate_monotone_cubic_bridge(freq,target,m,0,len(freq)-1)
    assert d["passed"]
    assert d["analytic_bridge_derivative_pass"]
    assert d["c1_slope_match"]


def test_endpoint_direction_is_not_repaired():
    with pytest.raises(ValueError):
        _evaluate_exact_c1_bridge(80.0,82.0,-1.0,2.0,.5)


def test_no_stable_handoff_fails_safe_to_base():
    freq=grid(); target=np.zeros_like(freq); m=np.ones_like(freq)
    out,d=adaptive_masked_handoff(freq,target,m)
    assert d["status"]=="NO_STABLE_HANDOFF"
    assert np.array_equal(out,target)


def test_deterministic():
    freq=grid(); target=base(freq); m=masked(freq)
    a,da=adaptive_masked_handoff(freq,target,m); b,db=adaptive_masked_handoff(freq,target,m)
    assert np.array_equal(a,b)
    assert da["transition_end_hz"]==db["transition_end_hz"]


def test_vertical_translation_invariance():
    freq=grid(); target=base(freq); m=masked(freq)
    a,da=adaptive_masked_handoff(freq,target,m); b,db=adaptive_masked_handoff(freq,target+37.25,m+37.25)
    assert da["transition_end_hz"]==db["transition_end_hz"]
    assert np.allclose(b,a+37.25,rtol=0,atol=1e-10)


def test_positive_scale_invariance():
    freq=grid(); target=base(freq); m=masked(freq)
    a,da=adaptive_masked_handoff(freq,target,m); b,db=adaptive_masked_handoff(freq,target*3,m*3)
    assert da["transition_end_hz"]==db["transition_end_hz"]
    assert np.allclose(b,a*3,rtol=0,atol=1e-9)


def test_no_endpoint_slope_clipping_in_diagnostics():
    freq=grid(); target=base(freq); m=masked(freq)
    out,d=adaptive_masked_handoff(freq,target,m)
    if d["status"]=="ADAPTIVE_HANDOFF":
        assert d["endpoint_target_slope_db_per_octave"]==d["endpoint_slope_start_used_db_per_octave"]
        assert d["endpoint_masked_slope_db_per_octave"]==d["endpoint_slope_end_used_db_per_octave"]

