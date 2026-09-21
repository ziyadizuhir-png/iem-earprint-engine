# engine/earprint_engine.py — FULL REPLACEMENT
# Purpose:
#   Preserve the existing Robust Mask / Masked EarPrint output.
#   Adaptive Handoff is a separate downstream transformation and must
#   NEVER replace or overwrite the Masked EarPrint representation.
#
# LOCKED OUTPUT SEMANTICS:
#   __mask.txt            = mask correction only
#   __robust_target.txt   = BaseTarget + Masked EarPrint correction
#   adaptive handoff      = downstream/on-demand output only
#
# IMPORTANT:
#   If Adaptive Handoff returns NO_STABLE_HANDOFF, __robust_target.txt
#   MUST remain the Masked EarPrint, not revert to BaseTarget.
#
# This file is intended as a replacement for the repository's current
# engine/earprint_engine.py while preserving the existing Robust Mask
# mathematics and configuration.

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("PyYAML is required.") from exc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "project.yaml"
PREFERRED_DIR = ROOT / "input" / "preferred"
TARGET_DIR = ROOT / "input" / "targets"
OUT = ROOT / "output"
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def fail(message: str) -> None:
    raise RuntimeError(message)


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = re.split(r"[\t,; ]+", line)

            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        fail(f"Not enough numeric XY data in {path}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)

    return unique_x, y[unique_idx]


def discover_txt(
    directory: Path,
    pattern: str,
    ignore_files: Iterable[str] | None = None,
) -> list[Path]:

    ignored = set(ignore_files or [])

    files = [
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.name not in ignored
    ]

    files.sort(key=lambda p: p.name.lower())

    return files


def interpolate_log_frequency(
    source_freq: np.ndarray,
    source_values: np.ndarray,
    target_freq: np.ndarray,
) -> np.ndarray:

    if np.any(source_freq <= 0) or np.any(target_freq <= 0):
        fail("Log-frequency interpolation requires positive frequencies.")

    return np.interp(
        np.log(target_freq),
        np.log(source_freq),
        source_values,
    )


def write_xy(
    path: Path,
    freq: np.ndarray,
    values: np.ndarray,
    decimals: int,
) -> None:

    with path.open("w", encoding="utf-8") as fh:
        for f, y in zip(freq, values):
            fh.write(
                f"{float(f):.12g}\t{float(y):.{decimals}f}\n"
            )


def read_output_xy(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None

    try:
        return parse_xy(path)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------

def gaussian_once_strict_domain(
    values: np.ndarray,
    freq: np.ndarray,
    domain_start_hz: float,
    domain_end_hz: float,
    sigma_indices: int,
    radius_indices: int,
) -> np.ndarray:

    # Preserve the existing strict-domain Gaussian concept:
    # smooth only inside the requested personal domain and leave
    # the rest unchanged.

    out = np.asarray(values, dtype=float).copy()

    mask = (
        (freq >= domain_start_hz)
        & (freq <= domain_end_hz)
    )

    idx = np.where(mask)[0]

    if len(idx) < 3:
        return out

    sigma = float(sigma_indices)
    radius = int(radius_indices)

    if sigma <= 0 or radius <= 0:
        return out

    offsets = np.arange(-radius, radius + 1, dtype=float)

    kernel = np.exp(
        -0.5 * (offsets / sigma) ** 2
    )

    kernel /= np.sum(kernel)

    source = out.copy()

    first = int(idx[0])
    last = int(idx[-1])

    for i in idx:
        rel = i - first

        lo = max(0, rel - radius)
        hi = min(last - first, rel + radius)

        local_rel = np.arange(lo, hi + 1)

        weights = kernel[
            local_rel - rel + radius
        ]

        weights = weights / np.sum(weights)

        out[i] = float(
            np.sum(
                source[first + local_rel]
                * weights
            )
        )

    return out


# ---------------------------------------------------------------------
# Boundary taper
# ---------------------------------------------------------------------

def sin2_boundary_taper(
    freq: np.ndarray,
    domain_start_hz: float,
    domain_end_hz: float,
    lower_transition_octaves: float,
    upper_transition_octaves: float,
) -> np.ndarray:

    out = np.zeros_like(freq, dtype=float)

    if lower_transition_octaves <= 0:
        lower_transition_octaves = 1e-12

    if upper_transition_octaves <= 0:
        upper_transition_octaves = 1e-12

    inside = (
        (freq >= domain_start_hz)
        & (freq <= domain_end_hz)
    )

    out[inside] = 1.0

    lower = (
        inside
        & (
            freq
            <
            domain_start_hz
            * 2.0 ** lower_transition_octaves
        )
    )

    if np.any(lower):
        t = (
            np.log2(freq[lower] / domain_start_hz)
            / lower_transition_octaves
        )

        t = np.clip(t, 0.0, 1.0)

        out[lower] = (
            np.sin(
                0.5 * math.pi * t
            ) ** 2
        )

    upper = (
        inside
        & (
            freq
            >
            domain_end_hz
            / 2.0 ** upper_transition_octaves
        )
    )

    if np.any(upper):
        t = (
            np.log2(domain_end_hz / freq[upper])
            / upper_transition_octaves
        )

        t = np.clip(t, 0.0, 1.0)

        out[upper] = (
            np.sin(
                0.5 * math.pi * t
            ) ** 2
        )

    return out


# ---------------------------------------------------------------------
# Huber robust consensus
# ---------------------------------------------------------------------

def huber_consensus(
    stack: np.ndarray,
    tuning_constant: float,
    scale_factor: float,
    scale_floor_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    x = np.asarray(stack, dtype=float)

    if x.ndim != 2:
        fail("Huber consensus expects a 2-D stack.")

    centre0 = np.median(x, axis=0)

    residual = x - centre0[None, :]

    mad = np.median(
        np.abs(residual),
        axis=0,
    )

    scale = np.maximum(
        scale_factor * mad,
        scale_floor_db,
    )

    u = residual / scale[None, :]

    weights = np.ones_like(u)

    large = np.abs(u) > tuning_constant

    weights[large] = (
        tuning_constant
        / np.abs(u[large])
    )

    numerator = np.sum(
        weights * x,
        axis=0,
    )

    denominator = np.sum(
        weights,
        axis=0,
    )

    centre = numerator / denominator

    return centre, scale, weights


# ---------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------

def alignment_scenarios(
    master_freq: np.ndarray,
    curve: np.ndarray,
    reference: np.ndarray,
    bands: list[tuple[float, float]],
):
    offsets = []

    scenario_curves = []

    for lo, hi in bands:
        mask = (
            (master_freq >= lo)
            & (master_freq <= hi)
        )

        if not np.any(mask):
            fail(
                f"Alignment band {lo}-{hi} Hz "
                "does not overlap master grid."
            )

        offset = float(
            np.median(
                curve[mask]
                - reference[mask]
            )
        )

        offsets.append(offset)

        scenario_curves.append(
            curve - offset
        )

    scenario_curves = np.vstack(
        scenario_curves
    )

    scenario_centre = np.median(
        scenario_curves,
        axis=0,
    )

    scenario_unc = np.median(
        np.abs(
            scenario_curves
            - scenario_centre[None, :]
        ),
        axis=0,
    )

    return (
        np.asarray(offsets, dtype=float),
        scenario_curves,
        scenario_centre,
        scenario_unc,
    )


# ---------------------------------------------------------------------
# Adaptive Handoff import
# ---------------------------------------------------------------------

def _load_adaptive_handoff():
    try:
        from engine.adaptive_handoff import (
            adaptive_masked_handoff,
        )

        return adaptive_masked_handoff

    except Exception:
        return None


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------

def main() -> None:

    cfg = read_config()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery = cfg["discovery"]
    frequency_cfg = cfg["frequency"]
    alignment_cfg = cfg["alignment"]
    smoothing_cfg = cfg["smoothing"]
    robust_cfg = cfg["robust_mask"]
    adaptive_cfg = cfg.get(
        "adaptive_handoff",
        {},
    )
    output_cfg = cfg["output"]

    huber_cfg = robust_cfg.get(
        "huber",
        {},
    )

    huber_tuning_constant = float(
        huber_cfg.get(
            "tuning_constant",
            robust_cfg.get(
                "tuning_constant",
                1.345,
            ),
        )
    )

    huber_scale_factor = float(
        huber_cfg.get(
            "scale_factor",
            robust_cfg.get(
                "scale_factor",
                1.4826,
            ),
        )
    )

    huber_scale_floor_db = float(
        huber_cfg.get(
            "scale_floor_db",
            robust_cfg.get(
                "scale_floor_db",
                0.15,
            ),
        )
    )

    huber_passes = int(
        huber_cfg.get(
            "passes",
            robust_cfg.get(
                "passes",
                1,
            ),
        )
    )

    if huber_passes != 1:
        fail(
            "This engine implements exactly one Huber reweighting pass."
        )

    bands = [
        tuple(map(float, p))
        for p in alignment_cfg["bands_hz"]
    ]

    personal_start = float(
        frequency_cfg["personal_start_hz"]
    )

    personal_end = float(
        frequency_cfg["personal_end_hz"]
    )

    output_start = float(
        frequency_cfg["output_start_hz"]
    )

    output_end = float(
        frequency_cfg["output_end_hz"]
    )

    sigma = int(
        smoothing_cfg["sigma_indices"]
    )

    radius = int(
        smoothing_cfg["radius_indices"]
    )

    decimals = int(
        output_cfg["decimals"]
    )

    boundary_taper_cfg = robust_cfg.get(
        "boundary_taper",
        {},
    )

    lower_transition_octaves = float(
        boundary_taper_cfg.get(
            "lower_transition_octaves",
            1 / 3,
        )
    )

    upper_transition_octaves = float(
        boundary_taper_cfg.get(
            "upper_transition_octaves",
            1 / 4,
        )
    )

    preferred_files = discover_txt(
        PREFERRED_DIR,
        discovery.get(
            "preferred_glob",
            "*.txt",
        ),
        discovery.get(
            "ignore_files",
            [],
        ),
    )

    target_files = discover_txt(
        TARGET_DIR,
        discovery.get(
            "target_glob",
            "*.txt",
        ),
        discovery.get(
            "ignore_files",
            [],
        ),
    )

    preferred = {
        p.name: parse_xy(p)
        for p in preferred_files
    }

    targets = {
        p.name: parse_xy(p)
        for p in target_files
    }

    lf_reference_file = cfg[
        "low_frequency_reference"
    ]["file"]

    if lf_reference_file not in targets:
        fail(
            f"Configured low-frequency reference "
            f"{lf_reference_file!r} was not discovered "
            "in input/targets."
        )

    master_freq, master_lf = targets[
        lf_reference_file
    ]

    if master_freq[0] > output_start:
        fail(
            "Low-frequency reference does not cover configured output start."
        )

    if master_freq[-1] < min(
        personal_end,
        output_end,
    ):
        fail(
            "Low-frequency reference does not cover required domain."
        )

    preferred_on_master = {}

    for path in preferred_files:
        f, y = preferred[path.name]

        preferred_on_master[path.name] = (
            interpolate_log_frequency(
                f,
                y,
                master_freq,
            )
        )

    # ================================================================
    # PURE EARPRINT
    # ================================================================

    pure_aligned_curves = []
    alignment_report = []

    for path in preferred_files:

        (
            offsets,
            _scenario_curves,
            _scenario_centre,
            scenario_unc,
        ) = alignment_scenarios(
            master_freq,
            preferred_on_master[path.name],
            master_lf,
            bands,
        )

        selected_offset = float(
            np.median(offsets)
        )

        selected_aligned = (
            preferred_on_master[path.name]
            - selected_offset
        )

        pure_aligned_curves.append(
            selected_aligned
        )

        alignment_report.append({
            "iem": path.name,
            "selected_median_offset_db": selected_offset,
            "offset_200_1000_db": float(offsets[0]),
            "offset_200_500_db": float(offsets[1]),
            "offset_500_1000_db": float(offsets[2]),
            "scenario_uncertainty_median_db": float(
                np.median(scenario_unc)
            ),
            "scenario_uncertainty_max_db": float(
                np.max(scenario_unc)
            ),
        })

    pure_aligned_stack = np.vstack(
        pure_aligned_curves
    )

    (
        pure_huber_centre,
        _pure_huber_scale,
        _pure_huber_weights,
    ) = huber_consensus(
        pure_aligned_stack,
        tuning_constant=huber_tuning_constant,
        scale_factor=huber_scale_factor,
        scale_floor_db=huber_scale_floor_db,
    )

    pure_earprint_smoothing_sigma = 4
    pure_earprint_smoothing_radius = 16

    smoothed_personal = (
        gaussian_once_strict_domain(
            pure_huber_centre,
            master_freq,
            personal_start,
            personal_end,
            pure_earprint_smoothing_sigma,
            pure_earprint_smoothing_radius,
        )
    )

    idx_1k = int(
        np.argmin(
            np.abs(
                master_freq - personal_start
            )
        )
    )

    join_shift = float(
        master_lf[idx_1k]
        - smoothed_personal[idx_1k]
    )

    shifted_personal = (
        smoothed_personal
        + join_shift
    )

    pure = master_lf.copy()

    low_or_equal_1k = (
        master_freq <= personal_start
    )

    strict_personal = (
        (master_freq > personal_start)
        & (master_freq < personal_end)
    )

    pure[low_or_equal_1k] = (
        master_lf[low_or_equal_1k]
    )

    pure[strict_personal] = (
        shifted_personal[strict_personal]
    )

    hf = master_freq >= personal_end

    anchor_idx = np.where(
        master_freq < personal_end
    )[0][-1]

    anchor_freq = float(
        master_freq[anchor_idx]
    )

    anchor_level = float(
        pure[anchor_idx]
    )

    pure[hf] = (
        anchor_level
        + float(
            cfg["high_frequency_extension"][
                "slope_db_per_octave"
            ]
        )
        * np.log2(
            master_freq[hf]
            / anchor_freq
        )
    )

    out_mask = (
        (master_freq >= output_start)
        & (master_freq <= output_end)
    )

    output_freq = master_freq[out_mask]

    write_xy(
        OUT / "pure_earprint_dynamic.txt",
        output_freq,
        pure[out_mask],
        decimals,
    )

    # ================================================================
    # TARGET LOOP
    # ================================================================

    robust_statistics = []

    adaptive_reports = []

    adaptive_handoff = _load_adaptive_handoff()

    for target_path in target_files:

        target_name = target_path.name

        stem = safe_stem(target_name)

        tf, tv = targets[target_name]

        base_target = interpolate_log_frequency(
            tf,
            tv,
            master_freq,
        )

        # ------------------------------------------------------------
        # Per-IEM target deltas
        # ------------------------------------------------------------

        per_iem_delta = []
        per_iem_alignment_unc = []

        for iem_path in preferred_files:

            preferred_curve = (
                preferred_on_master[
                    iem_path.name
                ]
            )

            (
                _offsets,
                _scenario_curves,
                scenario_centre,
                scenario_unc,
            ) = alignment_scenarios(
                master_freq,
                preferred_curve,
                base_target,
                bands,
            )

            delta_i = (
                scenario_centre
                - base_target
            )

            per_iem_delta.append(
                delta_i
            )

            per_iem_alignment_unc.append(
                scenario_unc
            )

        delta_stack = np.vstack(
            per_iem_delta
        )

        alignment_unc_stack = np.vstack(
            per_iem_alignment_unc
        )

        (
            centre,
            huber_scale,
            huber_weights,
        ) = huber_consensus(
            delta_stack,
            tuning_constant=huber_tuning_constant,
            scale_factor=huber_scale_factor,
            scale_floor_db=huber_scale_floor_db,
        )

        cross_iem_mad = np.median(
            np.abs(
                delta_stack
                - np.median(
                    delta_stack,
                    axis=0,
                )[None, :]
            ),
            axis=0,
        )

        alignment_uncertainty = np.median(
            alignment_unc_stack,
            axis=0,
        )

        # ------------------------------------------------------------
        # ROBUST MASK
        # ------------------------------------------------------------

        raw_mask = centre.copy()

        mask = gaussian_once_strict_domain(
            raw_mask,
            master_freq,
            personal_start,
            personal_end,
            sigma,
            radius,
        )

        mask *= sin2_boundary_taper(
            master_freq,
            personal_start,
            personal_end,
            lower_transition_octaves=(
                lower_transition_octaves
            ),
            upper_transition_octaves=(
                upper_transition_octaves
            ),
        )

        outside_personal = (
            (master_freq <= personal_start)
            | (master_freq >= personal_end)
        )

        mask[outside_personal] = 0.0

        # ------------------------------------------------------------
        # MASKED EARPRINT — THIS IS THE LOCKED ROBUST REPRESENTATION
        # ------------------------------------------------------------
        #
        # This variable MUST NOT be replaced by Adaptive Handoff.
        #
        masked_target = (
            base_target + mask
        )

        masked_target[outside_personal] = (
            base_target[outside_personal]
        )

        # ------------------------------------------------------------
        # WRITE MASK FIRST
        # ------------------------------------------------------------

        if output_cfg.get(
            "write_masks",
            True,
        ):
            write_xy(
                OUT / f"{stem}__mask.txt",
                output_freq,
                mask[out_mask],
                decimals,
            )

        # ------------------------------------------------------------
        # WRITE ROBUST TARGET FROM MASKED EARPRINT ONLY
        # ------------------------------------------------------------
        #
        # This is the critical architectural fix.
        #
        # __robust_target.txt represents:
        #
        #       BaseTarget + Mask
        #
        # It is NEVER populated from adaptive_handoff output.
        # ------------------------------------------------------------

        robust_target = masked_target.copy()

        if output_cfg.get(
            "write_robust_targets",
            True,
        ):
            write_xy(
                OUT / f"{stem}__robust_target.txt",
                output_freq,
                robust_target[out_mask],
                decimals,
            )

        # ------------------------------------------------------------
        # ADAPTIVE HANDOFF — DOWNSTREAM ONLY
        # ------------------------------------------------------------
        #
        # Adaptive Handoff receives the already-created Masked EarPrint.
        #
        # Its result is stored separately in memory/report data.
        #
        # It MUST NOT overwrite robust_target.
        # ------------------------------------------------------------

        handoff_report = {
            "status": "DISABLED",
            "anchor_hz": personal_start,
            "handoff_hz": personal_start,
            "endpoint_hz": personal_start,
            "transition_octaves": 0.0,
        }

        adaptive_output = None

        if adaptive_cfg.get(
            "enabled",
            False,
        ):

            if adaptive_handoff is None:

                handoff_report = {
                    "status": "UNAVAILABLE",
                    "anchor_hz": float(
                        adaptive_cfg.get(
                            "nominal_anchor_hz",
                            personal_start,
                        )
                    ),
                    "handoff_hz": None,
                    "endpoint_hz": None,
                    "transition_octaves": None,
                    "reason": (
                        "engine.adaptive_handoff "
                        "could not be imported."
                    ),
                }

            else:

                adaptive_output, handoff_report = (
                    adaptive_handoff(
                        master_freq,
                        base_target,
                        masked_target,
                        nominal_hz=float(
                            adaptive_cfg.get(
                                "nominal_anchor_hz",
                                personal_start,
                            )
                        ),
                        domain_end_hz=float(
                            personal_end
                        ),
                        min_transition_octaves=float(
                            adaptive_cfg.get(
                                "min_transition_octaves",
                                1.0 / 3.0,
                            )
                        ),
                        max_transition_octaves=float(
                            adaptive_cfg.get(
                                "max_transition_octaves",
                                0.8,
                            )
                        ),
                        stability_window_octaves=float(
                            adaptive_cfg.get(
                                "stability_window_octaves",
                                0.20,
                            )
                        ),
                    )
                )

        # ------------------------------------------------------------
        # HARD SAFETY ASSERTION
        # ------------------------------------------------------------
        #
        # Adaptive Handoff is not allowed to alter the Robust Target.
        #
        if not np.allclose(
            robust_target,
            masked_target,
            rtol=0.0,
            atol=1e-12,
        ):
            fail(
                "ARCHITECTURE ERROR: robust_target diverged "
                "from masked_target."
            )

        # ------------------------------------------------------------
        # OPTIONAL SEPARATE ADAPTIVE OUTPUT
        # ------------------------------------------------------------
        #
        # We deliberately do NOT write this to __robust_target.txt.
        #
        # If the repository later adds a dedicated adaptive output
        # filename, it should be written here. Until then, the adaptive
        # result remains a downstream computation/report result.
        # ------------------------------------------------------------

        adaptive_reports.append({
            "target": target_name,
            **handoff_report,
        })

        robust_statistics.append({
            "target": target_name,
            "huber_tuning_constant": huber_tuning_constant,
            "huber_scale_factor": huber_scale_factor,
            "huber_scale_floor_db": huber_scale_floor_db,
            "cross_iem_mad_median_db": float(
                np.median(cross_iem_mad)
            ),
            "huber_scale_median_db": float(
                np.median(huber_scale)
            ),
            "huber_downweighted_point_fraction": float(
                np.mean(
                    huber_weights < 1.0
                )
            ),
            "alignment_uncertainty_median_db": float(
                np.median(alignment_uncertainty)
            ),
            "alignment_uncertainty_max_db": float(
                np.max(alignment_uncertainty)
            ),
            "repeatability_floor_median_db": 0.0,
            "mask_abs_max_db": float(
                np.max(np.abs(mask))
            ),
            "mask_rms_db": float(
                np.sqrt(
                    np.mean(mask ** 2)
                )
            ),
            "adaptive_handoff_status": (
                handoff_report.get("status")
            ),
            "adaptive_handoff_hz": (
                handoff_report.get("handoff_hz")
            ),
            "adaptive_endpoint_hz": (
                handoff_report.get("endpoint_hz")
            ),
            "adaptive_transition_octaves": (
                handoff_report.get(
                    "transition_octaves"
                )
            ),
        })

    # ================================================================
    # REPORTS
    # ================================================================

    with (
        REPORTS / "alignment_report.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:

        if alignment_report:

            writer = csv.DictWriter(
                fh,
                fieldnames=list(
                    alignment_report[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                alignment_report
            )

    with (
        REPORTS / "robust_statistics.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:

        if robust_statistics:

            writer = csv.DictWriter(
                fh,
                fieldnames=list(
                    robust_statistics[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                robust_statistics
            )

    with (
        REPORTS / "adaptive_handoff_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            adaptive_reports,
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    manifest = {
        "target_count": len(target_files),
        "targets": [
            p.name
            for p in target_files
        ],
        "preferred_count": len(preferred_files),
        "preferred": [
            p.name
            for p in preferred_files
        ],
        "low_frequency_reference": (
            lf_reference_file
        ),
        "personal_domain_hz": [
            personal_start,
            personal_end,
        ],
        "output_domain_hz": [
            output_start,
            output_end,
        ],
        "robust_mask_output_semantics": (
            "BaseTarget + Masked EarPrint correction"
        ),
        "adaptive_handoff_is_downstream": True,
        "adaptive_handoff_can_overwrite_robust_target": False,
        "adaptive_handoff_reports": (
            "reports/adaptive_handoff_report.json"
        ),
    }

    with (
        REPORTS / "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            manifest,
            fh,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()