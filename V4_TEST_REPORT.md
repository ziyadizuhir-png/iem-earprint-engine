# vNext4 LM/IRLS Test Report

Test environment: repository container, Node.js runtime and Python test suite. Browser/device runtime will vary.

## Static / syntax

- `node --check app/pudding_peq.js` — PASS
- No `__EARPRINT_PURE_CURVE__` access in the PEQ solver — PASS
- No `farCandidateAnchors` production bank — PASS
- Hybrid Builder user-facing element IDs — absent

## Deterministic PEQ regression

`node tests/test_peq_node.js` — PASS

Checks include:
- Huber weight behavior
- LM linear solve
- <=10-band constraint
- Fc/Gain/Q bounds
- quantization state
- LM/IRLS production path
- no legacy fallback
- perturbation stability metadata

## Full Pudding x Robust Target regression

`node tests/test_peq_targets_node.js` — PASS for all 6 generated Robust Targets.

Local observed results from the final implementation:

| Robust Target | Time* | Bands | RMSE dB | P95 dB | Max dB | Max Q | HF | Stability |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 5128 DF Tilt | ~2.9–4.5 s | 10 | 0.3073 | 0.4257 | 2.2350 | 2.81 | PASS | PASS |
| Headphones.com IEM DF Tilt | ~1.9–2.9 s | 10 | 0.2261 | 0.5268 | 1.5970 | 3.63 | PASS | PASS |
| IEF2025 | ~2.0–2.6 s | 9 | 0.3369 | 0.5104 | 2.4112 | 5.14 | PASS | PASS |
| JM-1 DF Tilt | ~1.1–1.3 s | 5 | 0.4314 | 0.9144 | 2.1991 | 2.10 | PASS | PASS |
| Listener Type 3.3 JM-1 | ~2.1–2.3 s | 10 | 0.2875 | 0.5665 | 2.0471 | 3.22 | PASS | PASS |
| SoundGuys | ~3.6–4.0 s | 9 | 0.4423 | 1.0448 | 4.2216 | 2.66 | PASS | PASS |

\*Local Node execution time only; it is not a guaranteed browser/device latency.

SoundGuys requires the final 12–20 kHz safety rescue. The safer exported solution deliberately accepts a higher in-domain RMSE rather than violating the HF guard.

No target used the legacy numerical fallback.

## Python / locked engine

`python3 -m pytest -q` — **53 passed, 2 skipped**.

A clean-copy run of `python3 engine/earprint_engine.py` reported:

- BUILD PASS
- 8 independent IEM votes discovered
- 6 targets discovered
- 481 master-grid points
- 13 generated TXT outputs
- Specification math lock: PASS

`python3 -m pytest -q tests/test_regression_outputs.py` — **6 passed**.

## Speed comparison note

The previous vNext3 default JM-1 PEQ benchmark did not complete within a 120-second timeout in the same local container test. The vNext4 LM/IRLS JM-1 run completes in roughly 1–1.3 seconds in the same style of Node benchmark. This is a local engineering benchmark, not a promise of identical timing on phones/browsers.

## Remaining limitations

- RBJ sample rate remains provisional at 48 kHz as in the existing architecture.
- A legacy constrained-solver fallback remains available only for unexpected numerical edge cases and is explicitly reported if used.
- Real-device listening validation is still required because measurement fit metrics cannot by themselves establish subjective preference.
