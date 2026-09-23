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

## Pudding PEQ loss selection

The Pudding PEQ engine evaluates the existing standard loss and a robust Huber
loss (`delta = 1.0 dB`) as separate candidates. Production uses a balanced
workflow: both losses are screened at conservative Q 0.30–2.00, then only the
guarded winner receives the expensive high-Q rescue. Huber is committed only
when it improves RMSE without exceeding the P95 or maximum-error guard and when
the EarPrint Shape Guard passes. Otherwise the standard-loss result is retained.

This avoids running the high-Q rescue twice while keeping the same transactional
guards. The previous two-full-branch workflow remains available for audit by
setting `window.MoondropPuddingPEQ.CFG.performanceMode = 'exhaustive'` before
calling `optimize`.

Balanced mode uses a smaller search grid for the experimental Huber screen and
re-scores that candidate on the production grid before accepting it. Huber is
also skipped when the standard Q≤2 result is already within the robust trigger
(maximum error ≤ 3.0 dB and P95 ≤ 0.80 dB); this keeps ordinary runs responsive
without weakening the final Q10 or Shape Guard checks.

The conservative branch searches Q 0.30–2.00. A separate high-Q rescue tests
Q up to 10.00 and is committed only when its transactional guards pass. The
selected loss, guard decision, and both candidate summaries are included in
the exported JSON metadata.

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
