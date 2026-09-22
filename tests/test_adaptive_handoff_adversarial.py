import numpy as np
import pytest

import engine.adaptive_handoff as ah


def _grid():
    # Dense logarithmic grid containing the locked 1000 Hz anchor.
    return 1000.0 * 2.0 ** (np.arange(-48, 49, dtype=float) / 48.0)


def _simple_curves():
    f = _grid()
    x = np.log2(f / 1000.0)

    # Simple monotone curves. The test isolates candidate-selection semantics
    # by controlling only the destination-stability gate.
    base = 1.0 * x
    masked = 2.0 * x
    return f, base, masked


def _legal_candidates(f):
    return [
        i for i in range(len(f))
        if f[i] > 1000.0
        and np.log2(f[i] / 1000.0) >= 1.0 / 3.0 - 1e-12
        and np.log2(f[i] / 1000.0) <= 0.8 + 1e-12
    ]


def test_earliest_feasible_skips_first_infeasible_candidate(monkeypatch):
    """
    Strong adversarial test of the locked selection rule.

    The first legal E is forced to fail destination stability.
    The second legal E is forced to pass. Therefore the engine MUST select
    the second candidate, proving it searches for earliest FEASIBLE E rather
    than blindly taking the first legal candidate.
    """
    f, base, masked = _simple_curves()
    legal = _legal_candidates(f)
    assert len(legal) >= 2

    first_idx, second_idx = legal[:2]
    calls = []

    def fake_destination_stability(freq_hz, level_db, e_index, window):
        calls.append(e_index)
        if e_index == first_idx:
            return False, False, np.nan, np.nan
        return True, True, 2.0, 0.0

    monkeypatch.setattr(
        ah, "_destination_stability",
        fake_destination_stability,
    )

    out, diag = ah.adaptive_masked_handoff(f, base, masked)

    assert diag["status"] == "HANDOFF_ACCEPTED"
    assert diag["actual_handoff_hz"] == pytest.approx(f[second_idx])

    # The first candidate was actually evaluated and rejected.
    assert calls[0] == first_idx
    assert second_idx in calls
    assert np.all(np.isfinite(out))


def test_earliest_feasible_selects_first_passing_candidate(monkeypatch):
    """
    All legal candidates are declared stable. The engine MUST therefore
    select the first legal candidate.
    """
    f, base, masked = _simple_curves()
    legal = _legal_candidates(f)
    assert len(legal) >= 1

    calls = []

    def fake_destination_stability(freq_hz, level_db, e_index, window):
        calls.append(e_index)
        return True, True, 2.0, 0.0

    monkeypatch.setattr(
        ah, "_destination_stability",
        fake_destination_stability,
    )

    out, diag = ah.adaptive_masked_handoff(f, base, masked)

    assert diag["status"] == "HANDOFF_ACCEPTED"
    assert diag["actual_handoff_hz"] == pytest.approx(f[legal[0]])
    assert calls[0] == legal[0]
    assert np.all(np.isfinite(out))


def test_no_feasible_candidate_fails_safe(monkeypatch):
    """
    If every legal candidate fails the stability gate, the locked fail-safe
    must return the original BaseTarget.
    """
    f, base, masked = _simple_curves()

    def fake_destination_stability(freq_hz, level_db, e_index, window):
        return False, False, np.nan, np.nan

    monkeypatch.setattr(
        ah,
        "_destination_stability",
        fake_destination_stability,
    )

    out, diag = ah.adaptive_masked_handoff(f, base, masked)

    assert diag["status"] == "NO_STABLE_HANDOFF"
    assert diag["fail_safe"] == "BaseTarget"
    assert np.array_equal(out, base)
