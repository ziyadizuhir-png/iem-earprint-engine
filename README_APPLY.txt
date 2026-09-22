# EarPrint TRUE-C1 Full CI Fix

Overwrite the corresponding files in the existing repository.

## Files
- engine/adaptive_handoff.py
- tests/test_adaptive_handoff.py
- project.yaml
- config/project.yaml
- README_APPLY.txt

## CI failure fixed
Existing `engine/earprint_engine.py` calls:
`adaptive_masked_handoff(..., nominal_hz=...)`

The adaptive handoff now accepts `nominal_hz` as a compatibility alias while
retaining the locked 1000 Hz nominal anchor.

## Additional implementation fixes
- Hermite validation samples are no longer inserted into the production grid.
- The exact cubic is evaluated on the actual H→E production grid.
- Destination stability is checked only in the post-E stability window.
- Stability checks are normalized to preserve positive vertical-scale invariance.
- Backward-compatible bridge aliases are retained.

## Do NOT upload
- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`

## Locked architecture remains unchanged
TRUE-C1, raw endpoint slopes, alpha/beta gate, analytic derivative gate,
earliest feasible E, 1/3–0.8 octave bounds, destination stability gates,
and NO_STABLE_HANDOFF → BaseTarget fail-safe.

Run `pytest -q` after copying.
