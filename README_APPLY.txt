# EarPrint TRUE-C1 Hardening Update

Apply these files to `ziyadizuhir-png/iem-earprint-engine`.

## Files
- `engine/adaptive_handoff.py` — cleaned TRUE-C1 production implementation.
- `tests/test_adaptive_handoff.py` — hardened acceptance/invariance tests.
- `project.yaml` and `config/project.yaml` — locked architecture configuration.
- `README_APPLY.txt` — this file.

## Locked behavior preserved
- 1000 Hz nominal anchor.
- Minimum 1/3 octave transition width.
- Maximum 0.8 octave transition width.
- Earliest feasible E.
- Exact-C1 cubic Hermite bridge.
- Raw endpoint slopes retained exactly.
- Alpha/beta monotonicity condition.
- Analytic derivative gate.
- Destination slope/curvature stability gates.
- BaseTarget fail-safe on `NO_STABLE_HANDOFF`.
- No candidate scoring, Pareto selection, curvature optimization, target averaging, or target-specific optimization.

Run:
`pytest -q`

This package is a hardening update, not a change of tonal philosophy.
