# Runtime Solver Constraints

## 2026-09-29

- Added a validated runtime `solverConstraints` layer for maximum bands,
  minimum gain, and maximum gain.
- Kept hardware capability separate from user defaults: 10 bands and -12 to
  +10 dB hardware range versus 10 bands and -12 to +3 dB default user range.
- Propagated constraints through candidate generation, active-set growth,
  LM, Huber IRLS, high-Q rescue, redundancy/pruning, quantization, and exact
  RBJ export validation.
- Added responsive PEQ configuration controls and a live active-constraint
  summary to the dark audiophile UI.
- Added quantization sensitivity reporting, local worst-error regions,
  standardized diagnostics, high-Q rationale metadata, and deterministic
  solver-configuration hashes.
- Added Node regression coverage for band ceilings, gain ranges, invalid
  configurations, deterministic repeated solves, and capped growth schedules.
