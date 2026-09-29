# Pre-refinement repository audit

Audit performed before implementation changes for the EarPrint Engine + PEQ Engine refinement task.

## 1. Current file structure

```text
app/
  index.html                 Web UI and source loader
  pudding_peq.js             Single browser PEQ engine and report renderer
engine/
  earprint_engine.py         Single EarPrint build entry point and pipeline
  adaptive_handoff.py        Locked handoff implementation
  input_hash.py              Deterministic source hashing
  run_local.py               Local build wrapper
config/project.yaml           EarPrint and partial PEQ configuration
input/preferred/*.txt         Preferred/listener-adjusted IEM votes
input/original_711/*.txt      Source measurements used by the PEQ app
input/targets/*.txt           Target curves
output/*                      Generated EarPrint artifacts
reports/*                     Generated validation, statistics, and manifests
tests/*.py                    EarPrint and lifecycle tests
tests/*.js                    PEQ unit and production-data regressions
```

There is one EarPrint implementation (`engine/earprint_engine.py`) and one PEQ implementation (`app/pudding_peq.js`). No second engine or alternate production path was found.

## 2. EarPrint Engine flow

Entry point: `engine/earprint_engine.py:main`, called by `engine/run_local.py` and tests.

```text
discover input/preferred and input/targets
  -> parse_xy and coverage validation
  -> target reference supplies the master frequency grid
  -> log-frequency interpolation of preferred curves
  -> three low-frequency alignment scenarios per IEM
  -> median-selected alignment and per-IEM scenario uncertainty
  -> one-pass Huber consensus
  -> pure EarPrint smoothing and 1 kHz join / high-frequency extension
  -> for each target: aligned IEM deltas -> Huber robust mask
  -> target-mask smoothing and boundary taper
  -> optional locked adaptive handoff
  -> generated mask/robust-target/pure-EarPrint outputs and CSV/TXT reports
```

- Smoothing: `gaussian_once_strict_domain`; project smoothing is YAML-driven, but pure EarPrint uses hardcoded `sigma=4` and `radius=16`.
- Alignment: `alignment_scenarios` computes band medians, scenario centre, and max absolute scenario deviation.
- Consensus: `huber_consensus` uses median, MAD-derived scale, Huber cutoff, and weighted mean.
- Weighting: existing Huber weights only; alignment uncertainty is diagnostic and does not weight the consensus.
- Confidence: no EarPrint confidence-weighted consensus exists. The PEQ app has separate frequency/morphology confidence concepts.
- Output: dynamic filenames are generated from discovered targets; prior generated TXT/CSV/report files are cleared after input validation, so removed IEMs/targets do not leave stale generated artifacts.

## 3. PEQ Engine flow

Entry point: the browser UI in `app/index.html` loads `app/pudding_peq.js`; `optimize(rawCurve, targetCurve)` is the solver entry point.

```text
selected output/*__robust_target.txt
  -> parse and validate raw Pudding / target curves
  -> log-grid resampling and level alignment
  -> residual/error map and morphology feature analysis
  -> frequency confidence + Huber loss weighting
  -> evidence-driven candidate filters and active growth (3 -> 5 -> 7 -> <=10)
  -> joint LM refinement of log(Fc), gain, log(Q)
  -> optional Huber IRLS refinement with guarded acceptance
  -> response-aware pruning and filter merge/cleanup
  -> Q/high-Q/boundary and complexity penalties
  -> topology, exact RBJ, quantization, HF, and perturbation validation
  -> text/JSON/UI report generation
```

Existing components are all in `app/pudding_peq.js`: target loading, `objectiveComponents`, LM/IRLS optimizer, response-aware pruning/merge, Q control, smoothness/complexity/topology penalties, morphology confidence, and report rendering.

## 4. Existing feature check

| Feature | Current state |
|---|---|
| Huber weighting | Present in EarPrint Python and PEQ JavaScript; Python uses a tiny floating-point denominator guard, not the requested explicit epsilon. |
| Uncertainty calculation | Present as alignment scenario uncertainty and cross-IEM MAD diagnostics. |
| Confidence calculation | Present in PEQ as frequency confidence and residual morphology confidence; absent from EarPrint consensus weighting. |
| Smoothing parameters | Target/mask smoothing is in YAML; pure EarPrint sigma/radius are hardcoded. |
| Grid validation | Partial: positive/ascending/coverage checks exist; duplicate/non-finite/empty handling is not consistently centralized, and interpolation lacks explicit destination validation. |
| ISO226 support | Absent. |
| PEQ configuration | Partial YAML exists, but JavaScript has a separate hardcoded `CFG` and the documented values are not fully mirrored. |
| Report metrics | Present for Huber, uncertainty, PEQ error, topology, HF safety, stability perturbation, and headroom; no EarPrint stability score or ISO226 report fields. |

## 5. Dependency map

EarPrint:

```text
input/preferred + input/targets
  -> parse / validate / discover
  -> interpolate on target master grid
  -> alignment scenarios
  -> median-aligned curves / target deltas
  -> Huber consensus (pure EarPrint or robust mask)
  -> smoothing / taper / optional handoff
  -> output files and reports
```

PEQ:

```text
selected Robust Target + raw Pudding curve
  -> aligned error map
  -> existing frequency/morphology weighting and Huber loss
  -> existing LM/IRLS optimizer
  -> existing filter merge/prune and Q control
  -> topology/HF/quantization validation
  -> text/JSON/UI report
```

## 6. Detection results

- Duplicate engines: none found.
- Duplicate formulas: Huber-style loss exists in both engines because they are separate numerical layers; this is intentional architecture, not a duplicate EarPrint engine. PEQ objective terms are centralized in `objectiveComponents`.
- Duplicate constants: Huber defaults are repeated in Python fallback arguments/config access; pure EarPrint smoothing constants are hardcoded in code; PEQ limits and tuning values are duplicated between `config/project.yaml` and JavaScript `CFG`.
- Unused functions: no clearly dead production function was identified by the audit; exported PEQ test helpers are intentionally used by regression tests.
- Hardcoded tuning values: Python pure smoothing (`4`, `16`), Python fallback Huber defaults, JavaScript PEQ limits/tuning, report display limits, and several validation constants are hardcoded outside YAML.

## 7. Baseline constraints recorded

The master output grid is sourced from the configured low-frequency reference target and is preserved in generated output. The locked adaptive handoff, 1 kHz join, 1–12 kHz personal domain, output filenames, and existing API entry points are established baseline behavior and must remain unchanged.
