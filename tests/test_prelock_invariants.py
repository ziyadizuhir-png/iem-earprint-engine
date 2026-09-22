import numpy as np
import pytest
from engine.adaptive_handoff import adaptive_masked_handoff, _evaluate_exact_c1_bridge

def curves():
    f=np.geomspace(500.0,5000.0,161)
    x=np.log2(f/1000.0)
    return f,0.8*x,1.8*x+0.15*np.tanh(2*x)

def test_locked_bounds():
    f,b,m=curves()
    with pytest.raises(ValueError): adaptive_masked_handoff(f,b,m,min_transition_octaves=.32)
    with pytest.raises(ValueError): adaptive_masked_handoff(f,b,m,max_transition_octaves=.81)

def test_translation_invariance():
    f,b,m=curves(); o1,d1=adaptive_masked_handoff(f,b,m); o2,d2=adaptive_masked_handoff(f,b+7,m+7)
    assert d1["status"]==d2["status"]
    if d1["status"]=="HANDOFF_ACCEPTED": assert d1["actual_handoff_hz"]==pytest.approx(d2["actual_handoff_hz"])
    assert np.allclose(o2,o1+7)

def test_positive_scale_invariance():
    f,b,m=curves(); o1,d1=adaptive_masked_handoff(f,b,m); o2,d2=adaptive_masked_handoff(f,b*2,m*2)
    assert d1["status"]==d2["status"]
    if d1["status"]=="HANDOFF_ACCEPTED": assert d1["actual_handoff_hz"]==pytest.approx(d2["actual_handoff_hz"])
    assert np.allclose(o2,o1*2)

def test_anchor_required():
    f=np.geomspace(900,5000,80); b=np.zeros_like(f); m=np.ones_like(f)
    o,d=adaptive_masked_handoff(f,b,m)
    assert d["status"]=="NO_STABLE_HANDOFF" and np.array_equal(o,b)

def test_fail_safe():
    f=np.geomspace(500,5000,161); b=np.zeros_like(f); x=np.log2(f/1000); m=2-2*x
    o,d=adaptive_masked_handoff(f,b,m)
    assert d["status"]=="NO_STABLE_HANDOFF" and np.array_equal(o,b)

def test_exact_c1():
    v,d=_evaluate_exact_c1_bridge(0,1,1,1,1)
    assert np.isclose(d["used_start_slope"],1) and np.isclose(d["used_end_slope"],1)
    assert d["bridge_pass"] is True and v[0]==0 and v[-1]==1

def test_alpha_beta_gate():
    with pytest.raises(ValueError): _evaluate_exact_c1_bridge(0,1,3.5,3.5,1)

def test_deterministic_repeat():
    f,b,m=curves(); o1,d1=adaptive_masked_handoff(f,b,m); o2,d2=adaptive_masked_handoff(f,b,m)
    assert d1==d2 and np.array_equal(o1,o2)
