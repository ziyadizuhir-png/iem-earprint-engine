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
→ runtime solver constraints (user ceiling + gain range)
→ active band growth (3 → 5 → 7 → <= user ceiling)
→ joint Levenberg-Marquardt refinement of log(Fc), Gain, log(Q)
→ Huber IRLS robust refinement with guarded acceptance
→ response-aware pruning / redundancy cleanup
→ exact RBJ dense-grid validation
→ export quantization + local quantization rescue
→ 12–20 kHz safety guard
→ perturbation stability check
→ final PEQ export
```

The hardware capability is 10 bands and -12 to +10 dB. The default user solver
constraint is 10 bands and -12 to +3 dB. These are separate: the user limits are
validated once and propagated through candidate generation, active-set growth,
LM, Huber IRLS, rescue/pruning, quantization, and final exact-RBJ validation.
The engine does not generate an oversized bank and slice it afterward. All fitting and validation
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

### Pudding and WalkPlay workflows

The app exposes two workflow tabs over the same solver and Robust Target list:

- **Pudding** automatically loads `input/original_711/moondrop pudding fr.txt`.
  Its Moondrop Link constraints are fixed at 10 bands, -12 to +3 dB gain, and
  Q 0.30–10.00. The controls are read-only so the internal device contract is
  not accidentally changed.
- **WalkPlay** accepts a user FR upload in CSV, TXT, or JSON form. The input
  layer removes invalid points, sorts frequency, accepts small overshoot above
  20 kHz while normalizing the usable curve to 20 Hz–20 kHz, and reports the
  original/processed/removed counts. It supports 8 hardware bands by default
  or 10 bands, -10 to +10 dB gain, Q 0.10–5.00, and 0.1 dB gain rounding.

All raw IEM FR sources pass through the same defensive input normalizer. The
Robust Target path remains separate and locked. The normalizer also
handles future IEM files with blank/header rows, malformed or non-finite
values, duplicate frequencies, unsorted points, sub-20 Hz points, and points
above 20 kHz. These points are removed before log-frequency interpolation; the
solver only receives the resulting usable curve. The UI reports the cleanup
counts so a new IEM source does not require a code-specific exception.

The target dropdown is populated dynamically from READY Robust Target outputs;
the default selection is `5128 DF Tilt -1dB Oct B 4dB Target` when available,
while other discovered Robust Targets remain selectable. No alternate target
family is introduced.

Moondrop Link has no preamp. After the fixed Pudding limits are applied, the
maximum positive exported gain is measured and compensation is
`-min(maxBoost, 3 dB)`. The TXT export contains only frequency, gain, and Q
filters; the UI and JSON metadata show the compensation summary.

WalkPlay DAC volume is a playback recommendation rather than a solver input:
`recommended DAC = -(maximum PEQ boost + 1 dB)`, clamped to -8…+4 dB. The
recommendation preserves the PEQ curve and reflects the signal order PEQ → DAC
Playback Volume → output.

### FR input and device profiles

The selected target remains the dynamically loaded Robust Target only. Device
handling is post-processing after the solver:

- Generic PEQ preserves solver precision and reports a virtual preamp value.
- Moondrop Link applies capped automatic no-preamp gain compensation and
  rounds gain to 0.1 dB and Q to 0.01.
- WalkPlay / CrinEar DSP uses contribution-ranked band reduction and keeps the
  locked target and optimization objective unchanged.

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
