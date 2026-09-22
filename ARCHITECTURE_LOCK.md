# IEM EarPrint Engine — Architecture Lock

## Data roles
- `input/original_711/` = raw IEM measurement data.
- `input/preferred/` = listener-adjusted response data from the defined SoundOre sine-sweep + human PEQ procedure.
- `input/targets/` = external reference target frameworks.

A preferred-response file is not an IEM-selection preference and is not an anatomical hearing measurement. EarPrint is a robust consensus representation of the observed adjustment responses.

## Locked pipeline
```text
Listener-adjusted responses
→ log-frequency interpolation
→ alignment scenarios
→ per-response median scenario centre
→ one-pass reweighted Huber consensus
→ Pure EarPrint
→ target-specific alignment
→ one-pass Huber Robust Mask
→ boundary taper
→ Masked EarPrint
→ adaptive boundary handoff
→ Robust Target
```

## Adaptive handoff
```text
f <= H       → BaseTarget exactly
H < f <= E   → exact-C1 monotone cubic Hermite bridge
f > E        → Masked EarPrint exactly
```

`H = 1000 Hz`.

`1/3 octave <= log2(E/H) <= 0.8 octave`.

`E` is selected by `earliest_feasible_E`. There is no candidate scoring, Pareto selection, curvature optimisation, subjective target ranking, or forced transition.

If no candidate passes all mandatory gates, the fail-safe is `NO_STABLE_HANDOFF` and BaseTarget is retained.

## Exact-C1 bridge
The raw endpoint slopes are retained. The hard shape gate is:
```text
alpha >= 0
beta >= 0
alpha + beta <= 3
```

## Destination stability constants
Existing production acceptance constants are explicitly locked as:
```text
slope_variation / slope_scale <= 0.35
curvature_variation / curvature_scale <= 1.50
```
These are acceptance constants, not user-facing preferences.

## Invariance requirements
- vertical translation invariance
- positive amplitude-scale invariance
- deterministic output
- exact BaseTarget through H
- exact Masked EarPrint after E

## Non-goals
No target-family averaging/ranking, anatomical-hearing reconstruction, subjective candidate scoring, visual smoothness optimisation, arbitrary normalization, manual tonal edits, extra tilt, or forced handoff.
