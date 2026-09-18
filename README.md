# IEM EarPrint Engine — Dynamic

A deterministic Python implementation of the EarPrint workflow designed so that the **data directory is the source of truth**.

## The important change: everything is dynamic

You do **not** edit Python when adding or replacing an IEM or target.

### IEMs

Every:

```text
input/preferred/*.txt
```

is automatically discovered and treated as **one independent IEM vote**.

So if you have:

```text
Pudding.txt
Ceramics_Ultra.txt
Kato.txt
Origin.txt
```

the engine automatically uses all four.

Delete `Kato.txt` → it is no longer a vote.

Replace `Pudding.txt` → the next build automatically uses the new data.

### Targets

Every:

```text
input/targets/*.txt
```

is automatically discovered and processed independently.

You can add:

```text
targets/
├── IEF2025.txt
├── JM1_Robust.txt
├── SoundGuys.txt
├── My_New_Target.txt
└── Another_Target.txt
```

without changing Python or adding target names to a list.

The engine will automatically generate, for **every target**:

```text
<target>__mask.txt
<target>__robust_target.txt
<target>__plus_earprint.txt
```

This is the key design: **targets are data, not code.**

## One exception: LF reference

Pure EarPrint needs one declared low-frequency reference. Therefore:

```yaml
low_frequency_reference:
  file: IEF2025.txt
```

is deliberately configurable.

This does **not** restrict target discovery. Every target is still processed independently.

If you want another target to be the LF reference, change only that one filename in `config/project.yaml`.

## Current dataset

The repository package contains the currently supplied:

### Preferred-response votes

- Moondrop Pudding
- ROSESELSA Ceramics Ultra
- Samsung Galaxy Buds2 Pro
- TForce Yuan Li
- 7Hz Timeless
- Sony WF-1000XM4
- HIFIMAN Svanar Wireless Jr
- FOCUS Vocal

### Targets

- IEF Preference 2025
- JM-1 robust
- JM-1 B4
- SoundGuys
- Headphones.com IEM DF robust

These are simply the initial files. The engine does not depend on these names except for the configured LF reference.

## Deterministic rules

- One IEM = one independent vote.
- FR, PEQ and preferred response from the same IEM are not separate votes.
- Preferred-response curves are consumed directly.
- No averaging of PEQ gains.
- Frequency grids are interpolated in log-frequency space.
- Alignment scenarios: `[200,1000]`, `[200,500]`, `[500,1000]` Hz.
- Every target gets its **own** independent robust mask.
- No universal mask.
- No target curves are stitched/averaged together.
- No subjective AI tonal decisions.
- No final manual normalization or arbitrary gain cap.
- Exactly one Gaussian smoothing pass in strict `>1 kHz` and `<12 kHz`.
- Mask = 0 at/below 1 kHz and at/above 12 kHz.
- >=12 kHz is a display-only `-6 dB/octave` log-frequency continuation.

## Repository

```text
iem-earprint-engine/
├── .github/workflows/main.yml
├── config/project.yaml
├── engine/earprint_engine.py
├── input/
│   ├── preferred/
│   │   └── *.txt
│   └── targets/
│       └── *.txt
├── output/
├── reports/
├── requirements.txt
└── README.md
```

## Updating an IEM

Replace/add a two-column numeric TXT file:

```text
input/preferred/My_IEM.txt
```

No Python edit required.

## Updating/adding a target

Replace/add:

```text
input/targets/My_Target.txt
```

No Python edit required.

The next GitHub Actions build discovers it automatically.

## GitHub workflow

The workflow runs when:

- `input/**` changes
- `config/**` changes
- `engine/**` changes
- `requirements.txt` changes
- the workflow changes

It can also be started manually.

After the run:

**GitHub → Actions → completed run → Artifacts → `earprint-results`**

## Local run

```bash
python -m pip install -r requirements.txt
python engine/earprint_engine.py
```

Expected:

```text
BUILD PASS
Dynamic preferred votes discovered: 8
Dynamic targets discovered: 5
```

## Fail loudly

The engine intentionally stops if:

- an input file is missing
- fewer than 10 numeric rows exist
- frequency is non-positive
- frequencies are not strictly ascending
- source coverage cannot support the required interpolation
- the configured LF reference does not exist

This prevents a silent bad EarPrint.

## Future extensions

The deterministic core can later add:

- PEQ reconstruction and preferred-response verification
- repeatability files that never add votes
- automatic regression tests
- change reports between commits
- GitHub Pages interface
- automatic ZIP output
