import numpy as np

from engine.adaptive_handoff import adaptive_masked_handoff


def _grid():
    # 1/48-octave grid: exact 1000 Hz plus ample post-E stability samples.
    return 1000.0 * 2.0 ** (np.arange(-48, 49, dtype=float) / 48.0)


def _make_masked_with_bad_first_candidate():
    f = _grid()
    x = np.log2(f / 1000.0)
    base = 1.0 * x

    # Stable rising destination, with a narrow perturbation centred on the
    # first legal E candidate. This makes the first legal candidate fail
    # the destination/bridge hard gates while a later candidate recovers.
    masked = 2.0 * x
    first_e = 1000.0 * 2.0 ** (1.0 / 3.0)
    i = int(np.argmin(np.abs(f - first_e)))
    masked[i] += 2.5

    return f, base, masked, i


def test_earliest_feasible_not_earliest_candidate():
    f, base, masked, first_candidate = _make_masked_with_bad_first_candidate()

    out, diag = adaptive_masked_handoff(f, base, masked)

    assert diag["status"] == "HANDOFF_ACCEPTED"

    # The first legal candidate is intentionally invalid, so the engine must
    # continue searching and choose a later candidate that passes every gate.
    assert diag["actual_handoff_hz"] > f[first_candidate]

    # The selected candidate must remain inside the locked transition range.
    assert (
        np.log2(diag["actual_handoff_hz"] / 1000.0)
        <= 0.8 + 1e-12
    )

    assert np.all(np.isfinite(out))


def test_all_feasible_candidates_choose_first():
    f = _grid()
    x = np.log2(f / 1000.0)

    base = 1.0 * x
    masked = 1.5 * x

    out, diag = adaptive_masked_handoff(f, base, masked)

    assert diag["status"] == "HANDOFF_ACCEPTED"

    legal = np.flatnonzero(
        (f > 1000.0)
        & (np.log2(f / 1000.0) >= 1.0 / 3.0 - 1e-12)
        & (np.log2(f / 1000.0) <= 0.8 + 1e-12)
    )

    assert len(legal) > 0

    # Here the candidates are intentionally well behaved, so the first
    # candidate in the legal range should also be the first feasible one.
    assert diag["actual_handoff_hz"] == f[legal[0]]
    assert np.all(np.isfinite(out))
