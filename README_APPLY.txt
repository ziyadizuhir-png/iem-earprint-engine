# EarPrint TRUE-C1 Hardening Update — corrected

This package fixes the earlier hardening ZIP.

## Important fixes
1. The validated 257-point Hermite diagnostic curve is no longer written directly
   into the caller's target grid. The exact cubic polynomial is evaluated on the
   actual production frequency grid.
2. Destination stability is evaluated over the intended ± stability window around
   E rather than unintentionally extending to the end of the target.
3. Slope/curvature stability tolerances are scale-normalized so the locked positive
   scaling invariance is preserved.
4. Tests use an explicit 1000 Hz computational anchor where required.

## Locked architecture preserved
- 1000 Hz nominal anchor.
- Minimum 1/3 octave transition width.
- Maximum 0.8 octave transition width.
- Earliest feasible E.
- Exact-C1 cubic Hermite bridge.
- Raw endpoint slopes retained exactly.
- alpha >= 0, beta >= 0, alpha + beta <= 3.
- Analytic derivative gate.
- Destination slope and curvature stability hard gates.
- BaseTarget fail-safe on NO_STABLE_HANDOFF.
- No candidate scoring, Pareto selection, curvature optimization, target averaging,
  or target-specific optimization.

## Verification
Local pytest result: 11 passed.

Run:
`pytest -q`
