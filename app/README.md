# EarPrint Personal Web App

Static GitHub Pages UI for the EarPrint repository.

The page reads repository inputs/outputs through GitHub's public contents/raw endpoints. It contains no write token.

Features:
- Dynamic IEM and target counts.
- Robust Mask / Robust Target viewer and download for the selected target only.
- On-demand Hybrid Builder for any discovered target.
- Configurable join frequency, default 1000 Hz.
- Hybrid math matches the repository's locked `build_hybrid_curve()` definition.
