# FR Input and Device Profiles

## 2026-09-29

- Added default Moondrop FR and local CSV/TXT/JSON custom FR input.
- Added strict frequency/dB validation and log-grid interpolation for custom
  FR gaps.
- Kept Robust Target as the only target source.
- Added Generic PEQ, Moondrop Link, and WalkPlay / CrinEar DSP post-processing
  profiles.
- Added Moondrop no-preamp headroom compensation and 0.1 dB / 0.1 Q output
  rounding.
- Added WalkPlay 8/10-band selection, hardware gain/Q bounds, DAC volume
  clipping warnings, and contribution-ranked band reduction.
- Added FR and device profile regression coverage.
