# vNext4 LM/IRLS Full Solver Upgrade

## Production solver

`app/pudding_peq.js` now uses a real joint Levenberg-Marquardt (LM) solver with Huber IRLS refinement as the production path.

### Added

- Joint continuous optimization of `log(Fc)`, `Gain`, and `log(Q)`.
- Numerical exact-biquad Jacobian using central finite differences.
- Damped LM normal-equation solve with pivoting, rollback, lambda increase/decrease, step limits, and convergence checks.
- Huber IRLS robust refinement with guarded acceptance.
- Evidence-driven active band growth: 3 -> 5 -> 7 -> <=10.
- Band-growth early stop when the extra bands do not earn validated improvement.
- Bandwidth-derived Q candidate initialization.
- Response caching on each frequency grid.
- Fast response-aware pruning and near-duplicate cleanup.
- Gain-energy, high-Q, and 1 kHz boundary high-Q regularization.
- Exact perturbation stability validation.
- Exact final export quantization and local quantization rescue.
- 12–20 kHz HF safety rescue for filters close to the 12 kHz correction boundary.
- Explicit legacy constrained-solver fallback only for numerical failure; fallback is recorded in result metadata.

### Architecture locks preserved

- Selected Robust Target is the sole PEQ target.
- Pure EarPrint is not read by the PEQ solver.
- Robust Target generation is unchanged.
- No fixed far-anchor/vendor candidate bank is used.
- Maximum 10 active bands is enforced during solving.
- Gain is constrained to -12 to +3 dB and Q to 0.30 to 10.
- Correction/fitting domain is 20 Hz–12 kHz.
- 12–20 kHz is validation-only.
- Final response uses exact RBJ peaking-biquad simulation at the provisional 48 kHz sample rate.

## App / CI

- Updated browser cache-bust version to `2026-09-25.5-vNext4-LM-IRLS`.
- Hybrid Builder remains absent from the user-facing UI.
- Added full production-data Node regression across every READY Robust Target.
- Web-app CI now runs both the deterministic synthetic PEQ regression and the full Robust Target regression.
