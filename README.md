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
loss (`delta = 1.0 dB`) as separate candidates. Huber is committed only when
it improves RMSE without exceeding the P95 or maximum-error guard and when the
EarPrint Shape Guard passes. Otherwise the standard-loss result is retained.

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
