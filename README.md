# IEM EarPrint Engine

Deterministic implementation of the locked EarPrint architecture.

## Data interpretation

`input/preferred/*.txt` contains listener-adjusted response curves produced through a human PEQ adjustment procedure using a SoundOre sine sweep over approximately 1–12 kHz.

They are observed listener-adjustment data, not raw IEM measurements and not simply a list of IEMs the listener likes.

```text
Raw IEM FR
→ SoundOre 1–12 kHz sweep
→ human PEQ adjustment
→ listener-adjusted response
→ alignment
→ robust EarPrint consensus
```

`input/original_711/` contains source IEM measurements. `input/targets/` contains external target/reference frameworks.

EarPrint is not an anatomical hearing reconstruction.

## Pudding PEQ solver

The web PEQ engine consumes the selected **final Robust Target as its sole PEQ
target**. Pure EarPrint remains upstream-only in Robust Target generation and is
not read as a second target or hidden PEQ objective.

The production solver is `vNext4-LM-IRLS`:

```text
Raw Pudding 711 + selected Robust Target
→ level alignment
→ residual / feature evidence
→ bandwidth-derived candidate filters
→ active band growth (3 → 5 → 7 → <=10)
→ joint Levenberg-Marquardt refinement of log(Fc), Gain, log(Q)
→ Huber IRLS robust refinement with guarded acceptance
→ response-aware pruning / redundancy cleanup
→ exact RBJ dense-grid validation
→ export quantization + local quantization rescue
→ 12–20 kHz safety guard
→ perturbation stability check
→ final PEQ export
```

The 10-band hardware limit is part of the active-set solve; the engine does not
generate an oversized bank and slice it afterward. All fitting and validation
uses the exact RBJ peaking-biquad magnitude response at the provisional 48 kHz
sample rate. The correction domain is 20–12,000 Hz. The 12–20 kHz region is
validation-only and is never fabricated as personal EarPrint data.

Huber is implemented through iteratively reweighted least squares (IRLS). A
Huber-refined state is accepted only when its robust fitting cost improves and
RMSE/P95/maximum-error guards remain bounded. High-Q filters are penalized and
1 kHz boundary high-Q corrections receive an additional internal penalty.

Final Fc/Gain/Q values are quantized to the export grid and exactly re-simulated.
A deterministic local rescue may adjust adjacent quantization steps. Final
metadata records band-growth decisions, LM iteration counts, stability evidence,
HF safety, modeled headroom and whether a numerical fallback was required.

## Locked handoff

```text
<= 1000 Hz → BaseTarget
1000 Hz → E → exact-C1 monotone cubic Hermite bridge
> E        → Masked EarPrint
```

`E` is the earliest feasible candidate in the locked 1/3–0.8 octave range. If no candidate passes every mandatory gate, the engine returns `NO_STABLE_HANDOFF` and retains BaseTarget.

See `ARCHITECTURE_LOCK.md` and the math-locked specification/addendum.

## Build safety and CI

The engine validates every discovered preferred and target curve before touching
existing generated results. GitHub Actions snapshots `output/` and `reports/`
and restores them if generation fails. A separate web-app workflow checks
`app/pudding_peq.js` syntax whenever the app changes.

Builds also store a deterministic source hash in
`reports/build_input_hash.txt`. When inputs, configuration, engine code, or
locked specifications are unchanged and required artifacts still exist, the
workflow reuses the generated results instead of recalculating them.
