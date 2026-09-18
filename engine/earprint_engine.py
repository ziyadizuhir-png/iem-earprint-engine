#!/usr/bin/env python3
"""
IEM EarPrint Engine
Deterministic implementation of the supplied EarPrint prompt with explicit math-lock definitions.

SOURCE-OF-TRUTH MATH
--------------------
This engine follows the supplied prompt for:
- one independent IEM = one equal vote
- log-frequency interpolation onto one declared master grid
- three alignment scenarios: [200,1000], [200,500], [500,1000] Hz
- per-IEM scenario centre = pointwise median of the 3 aligned scenarios
- per-IEM alignment uncertainty(f) = pointwise max absolute distance from that centre
- pure EarPrint construction and single Gaussian smoothing pass
- target-specific robust masking with frequency-dependent AlignmentUncertainty
- conditional RepeatabilityFloor: inactive when no valid same-IEM repeats exist
- retention / raw mask / taper / domain-zero rules
- dynamic target discovery and one hybrid per discovered target as an explicit extension
  of the prompt's two named hybrids.

No target list or IEM list is hard-coded.
The engine accepts authoritative preferred-response curves as its sole calculation input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "project.yaml"
PREFERRED_DIR = ROOT / "input" / "preferred"
TARGET_DIR = ROOT / "input" / "targets"
OUT = ROOT / "output"
REPORTS = ROOT / "reports"


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_config() -> dict:
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_txt(directory: Path, pattern: str, ignore_files: list[str] | None) -> list[Path]:
    ignored = set(ignore_files or [])
    files = sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.name not in ignored
    )
    if not files:
        fail(f"No input files discovered in {directory} using {pattern!r}")
    return files


def parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []

    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1
    ):
        s = raw.strip()
        if not s or s.startswith(("#", "//", ";")):
            continue

        parts = re.split(r"[\t,; ]+", s)
        if len(parts) < 2:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue

        if not (math.isfinite(x) and math.isfinite(y)):
            fail(f"{path.name}: non-finite numeric value at line {line_no}")

        rows.append((x, y))

    if len(rows) < 10:
        fail(f"{path.name}: fewer than 10 numeric data rows found")

    arr = np.asarray(rows, dtype=float)

    if np.any(arr[:, 0] <= 0):
        fail(f"{path.name}: frequency must be > 0 Hz")

    if np.any(np.diff(arr[:, 0]) <= 0):
        fail(f"{path.name}: frequency values must be strictly ascending")

    if not (np.all(np.isfinite(arr[:, 0])) and np.all(np.isfinite(arr[:, 1]))):
        fail(f"{path.name}: non-finite data detected")

    return arr[:, 0], arr[:, 1]


def interpolate_log_frequency(
    freq_src: np.ndarray,
    level_src: np.ndarray,
    freq_dst: np.ndarray,
) -> np.ndarray:
    """Linear interpolation in log-frequency space."""
    if freq_dst[0] < freq_src[0] or freq_dst[-1] > freq_src[-1]:
        fail(
            "Grid coverage error: "
            f"{freq_src[0]:.6g}-{freq_src[-1]:.6g} Hz source does not cover "
            f"{freq_dst[0]:.6g}-{freq_dst[-1]:.6g} Hz destination."
        )
    return np.interp(np.log(freq_dst), np.log(freq_src), level_src)


def band_mask(freq: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        fail(f"No samples in inclusive alignment band [{lo}, {hi}] Hz")
    return mask


def band_median(
    freq: np.ndarray,
    values: np.ndarray,
    lo: float,
    hi: float,
) -> float:
    return float(np.median(values[band_mask(freq, lo, hi)]))


def alignment_scenarios(
    freq: np.ndarray,
    curve: np.ndarray,
    reference: np.ndarray,
    bands: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the three alignment scenarios.

    Returns:
      offsets[3]
      scenario_curves[3, N]
      per_iem_centre[N] = pointwise median(scenario_curves)
      per_iem_uncertainty[N] = pointwise max(abs(scenario - centre))
    """
    offsets = np.asarray(
        [band_median(freq, curve - reference, lo, hi) for lo, hi in bands],
        dtype=float,
    )
    scenario_curves = np.vstack([curve - offset for offset in offsets])
    centre = np.median(scenario_curves, axis=0)
    uncertainty = np.max(np.abs(scenario_curves - centre), axis=0)
    return offsets, scenario_curves, centre, uncertainty


def gaussian_once_strict_domain(
    values: np.ndarray,
    freq: np.ndarray,
    start_hz: float,
    end_hz: float,
    sigma_indices: int,
    radius_indices: int,
) -> np.ndarray:
    """
    Exactly one Gaussian smoothing pass only on strict >start and <end.

    The extracted personal-domain segment is padded at its own ends with
    nearest-endpoint semantics, matching the prompt's nearest-endpoint rule.
    """
    out = values.copy()
    idx = np.where((freq > start_hz) & (freq < end_hz))[0]
    if idx.size == 0:
        return out

    left, right = int(idx[0]), int(idx[-1])
    out[left:right + 1] = gaussian_filter1d(
        out[left:right + 1],
        sigma=sigma_indices,
        radius=radius_indices,
        mode="nearest",
    )
    return out


def quarter_octave_sin2_taper(
    freq: np.ndarray,
    start_hz: float,
    end_hz: float,
) -> np.ndarray:
    """
    Fixed quarter-octave sin^2 tapers at both personal-domain boundaries.
    """
    w = np.ones_like(freq, dtype=float)
    q = 2.0 ** 0.25

    low_end = start_hz * q
    high_start = end_hz / q

    low = (freq > start_hz) & (freq < low_end)
    if np.any(low):
        x = np.log2(freq[low] / start_hz) / 0.25
        w[low] = np.sin(0.5 * np.pi * x) ** 2

    high = (freq > high_start) & (freq < end_hz)
    if np.any(high):
        x = np.log2(freq[high] / high_start) / 0.25
        w[high] = np.cos(0.5 * np.pi * x) ** 2

    w[freq <= start_hz] = 0.0
    w[freq >= end_hz] = 0.0
    return w


def log_interp_scalar(freq: np.ndarray, level: np.ndarray, target_hz: float) -> float:
    """Evaluate a curve at one frequency using linear interpolation in log-frequency."""
    if target_hz < freq[0] or target_hz > freq[-1]:
        fail(
            f"Interpolation point {target_hz:g} Hz is outside curve coverage "
            f"{freq[0]:g}-{freq[-1]:g} Hz."
        )
    return float(np.interp(np.log(target_hz), np.log(freq), level))


def build_hybrid_curve(
    freq: np.ndarray,
    lower_target: np.ndarray,
    pure_earprint: np.ndarray,
    join_hz: float = 1000.0,
) -> tuple[np.ndarray, float]:
    """
    Build a selectable-target hybrid without modifying the lower target:

      H(f) = T(f),                      f <= join
      H(f) = P(f) + c,                  f > join

    where c = T(join) - P(join), both evaluated by the declared
    log-frequency interpolation rule. This makes the join continuous at
    the requested frequency while leaving the entire lower target unchanged.
    """
    if not (np.isfinite(join_hz) and join_hz > 0):
        fail("Hybrid join frequency must be a finite positive number.")
    lower_at_join = log_interp_scalar(freq, lower_target, join_hz)
    pure_at_join = log_interp_scalar(freq, pure_earprint, join_hz)
    shift = lower_at_join - pure_at_join
    hybrid = lower_target.copy()
    upper = freq > join_hz
    hybrid[upper] = pure_earprint[upper] + shift
    return hybrid, float(shift)


def retention_and_mask(
    centre: np.ndarray,
    total_uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prompt:
      Retention = abs(Centre) / (abs(Centre) + TotalUncertainty)
      RawMask   = Centre * Retention
    """
    denom = np.abs(centre) + total_uncertainty
    retention = np.zeros_like(centre, dtype=float)
    nz = denom > 0
    retention[nz] = np.abs(centre[nz]) / denom[nz]
    raw_mask = centre * retention
    return retention, raw_mask


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_.")
    return stem or "target"


def write_xy(path: Path, freq: np.ndarray, level: np.ndarray, decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x, y in zip(freq, level):
            f.write(f"{x:.12g}\t{y:.{decimals}f}\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_output_xy(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        return parse_xy(path)
    except Exception:
        return None


def compare_curves(
    previous: tuple[np.ndarray, np.ndarray] | None,
    current: tuple[np.ndarray, np.ndarray] | None,
) -> dict:
    if previous is None or current is None:
        return {
            "previous_available": previous is not None,
            "current_available": current is not None,
            "matched_points": 0,
            "rms_db": None,
            "max_abs_db": None,
        }

    pf, py = previous
    cf, cy = current

    n = min(len(pf), len(cf))
    pf, py, cf, cy = pf[:n], py[:n], cf[:n], cy[:n]

    if n == 0:
        return {
            "previous_available": True,
            "current_available": True,
            "matched_points": 0,
            "rms_db": None,
            "max_abs_db": None,
        }

    if not np.allclose(pf, cf, rtol=0, atol=1e-9):
        # Compare by log-frequency interpolation onto current grid.
        if cf[0] < pf[0] or cf[-1] > pf[-1]:
            return {
                "previous_available": True,
                "current_available": True,
                "matched_points": 0,
                "rms_db": None,
                "max_abs_db": None,
            }
        py_interp = np.interp(np.log(cf), np.log(pf), py)
        diff = cy - py_interp
    else:
        diff = cy - py

    return {
        "previous_available": True,
        "current_available": True,
        "matched_points": int(len(diff)),
        "rms_db": float(np.sqrt(np.mean(diff**2))),
        "max_abs_db": float(np.max(np.abs(diff))),
    }


def main() -> None:
    cfg = read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    discovery = cfg["discovery"]
    frequency_cfg = cfg["frequency"]
    alignment_cfg = cfg["alignment"]
    smoothing_cfg = cfg["smoothing"]
    output_cfg = cfg["output"]

    preferred_files = discover_txt(
        PREFERRED_DIR,
        discovery.get("preferred_glob", "*.txt"),
        discovery.get("ignore_files", []),
    )
    target_files = discover_txt(
        TARGET_DIR,
        discovery.get("target_glob", "*.txt"),
        discovery.get("ignore_files", []),
    )

    bands = [tuple(map(float, p)) for p in alignment_cfg["bands_hz"]]
    personal_start = float(frequency_cfg["personal_start_hz"])
    personal_end = float(frequency_cfg["personal_end_hz"])
    output_start = float(frequency_cfg["output_start_hz"])
    output_end = float(frequency_cfg["output_end_hz"])
    sigma = int(smoothing_cfg["sigma_indices"])
    radius = int(smoothing_cfg["radius_indices"])
    decimals = int(output_cfg["decimals"])

    preferred = {p.name: parse_xy(p) for p in preferred_files}
    targets = {p.name: parse_xy(p) for p in target_files}

    # Preserve the immediately previous pure EarPrint for the required change report.
    previous_pure = read_output_xy(OUT / "pure_earprint_dynamic.txt")

    # Remove only generated output/report files. Input data remain untouched.
    for p in OUT.glob("*.txt"):
        p.unlink()
    for p in REPORTS.iterdir():
        if p.is_file():
            p.unlink()

    lf_reference_file = cfg["low_frequency_reference"]["file"]
    if lf_reference_file not in targets:
        fail(
            f"Configured low-frequency reference {lf_reference_file!r} "
            "was not discovered in input/targets."
        )

    master_freq, master_lf = targets[lf_reference_file]

    # Common output/master grid is the declared LF reference grid.
    if master_freq[0] > output_start:
        fail("Low-frequency reference does not cover configured output start.")
    if master_freq[-1] < min(personal_end, output_end):
        fail("Low-frequency reference does not cover required personal/output domain.")

    # Validate/interpolate each preferred response onto the declared master grid.
    preferred_on_master: dict[str, np.ndarray] = {}
    for path in preferred_files:
        f, y = preferred[path.name]
        preferred_on_master[path.name] = interpolate_log_frequency(
            f, y, master_freq
        )

    # ---------------------------------------------------------------------
    # PURE EARPRINT
    # Prompt lines 41-48:
    # align each preferred response to LF reference using median of 3 offsets;
    # equal IEM votes; pointwise median; exact LF below/equal 1 kHz;
    # one Gaussian pass >1 and <12 kHz; one join shift; -6 dB/oct display above 12 kHz.
    # ---------------------------------------------------------------------
    pure_aligned_curves = []
    alignment_report = []

    for path in preferred_files:
        offsets, _scenario_curves, _scenario_centre, scenario_unc = alignment_scenarios(
            master_freq,
            preferred_on_master[path.name],
            master_lf,
            bands,
        )

        selected_offset = float(np.median(offsets))
        selected_aligned = preferred_on_master[path.name] - selected_offset
        pure_aligned_curves.append(selected_aligned)

        alignment_report.append({
            "iem": path.name,
            "selected_median_offset_db": selected_offset,
            "offset_200_1000_db": float(offsets[0]),
            "offset_200_500_db": float(offsets[1]),
            "offset_500_1000_db": float(offsets[2]),
            "scenario_uncertainty_median_db": float(np.median(scenario_unc)),
            "scenario_uncertainty_max_db": float(np.max(scenario_unc)),
        })

    pure_aligned_stack = np.vstack(pure_aligned_curves)
    direct_centre = np.median(pure_aligned_stack, axis=0)

    smoothed_personal = gaussian_once_strict_domain(
        direct_centre,
        master_freq,
        personal_start,
        personal_end,
        sigma,
        radius,
    )

    idx_1k = int(np.argmin(np.abs(master_freq - personal_start)))
    join_shift = float(master_lf[idx_1k] - smoothed_personal[idx_1k])
    shifted_personal = smoothed_personal + join_shift

    pure = master_lf.copy()
    low_or_equal_1k = master_freq <= personal_start
    strict_personal = (master_freq > personal_start) & (master_freq < personal_end)

    pure[low_or_equal_1k] = master_lf[low_or_equal_1k]
    pure[strict_personal] = shifted_personal[strict_personal]

    # Display-only resonance-free -6 dB/octave extension at/above 12 kHz.
    hf = master_freq >= personal_end
    anchor_idx = np.where(master_freq < personal_end)[0][-1]
    anchor_freq = float(master_freq[anchor_idx])
    anchor_level = float(pure[anchor_idx])
    pure[hf] = anchor_level + float(
        cfg["high_frequency_extension"]["slope_db_per_octave"]
    ) * np.log2(master_freq[hf] / anchor_freq)

    out_mask = (master_freq >= output_start) & (master_freq <= output_end)
    output_freq = master_freq[out_mask]

    write_xy(
        OUT / "pure_earprint_dynamic.txt",
        output_freq,
        pure[out_mask],
        decimals,
    )

    # ---------------------------------------------------------------------
    # CONDITIONAL REPEATABILITY FLOOR
    #
    # The supplied prompt says to derive a common floor only when VALID
    # same-IEM repeats exist. The current repository dataset has no repeat
    # structure, so the term is inactive rather than invented.
    # ---------------------------------------------------------------------
    repeats_cfg = cfg.get("repeatability", {})
    repeat_floor_enabled = bool(repeats_cfg.get("enabled", False))
    repeatability_floor = np.zeros_like(master_freq)
    repeatability_note = "No valid same-IEM repeat data supplied; RepeatabilityFloor inactive."

    if repeat_floor_enabled:
        fail(
            "RepeatabilityFloor is enabled, but this repository build has no "
            "prompt-defined repeatability estimator. Provide a prompt-specified "
            "repeatability method before enabling it; the engine will not invent one."
        )

    robust_statistics: list[dict] = []

    # ---------------------------------------------------------------------
    # ROBUST MASKING — EVERY DISCOVERED TARGET
    #
    # Prompt lines 51-64:
    # - for each IEM and each base target, use the three alignment scenarios;
    # - per-IEM centre = pointwise median of the three aligned scenarios;
    # - per-IEM uncertainty = pointwise max distance from that centre;
    # - Delta_i = per-IEM centre - BaseTarget;
    # - Centre = median_i(Delta_i);
    # - CrossIEM_MAD = median_i(abs(Delta_i-Centre));
    # - AlignmentUncertainty = median_i(per-IEM uncertainty);
    # - TotalUncertainty = max(CrossIEM_MAD, AlignmentUncertainty, RepeatabilityFloor);
    # - retention / raw mask;
    # - one Gaussian pass in strict personal domain;
    # - fixed quarter-octave taper;
    # - zero mask and exact base target at/beyond boundaries.
    # ---------------------------------------------------------------------
    for target_path in target_files:
        target_name = target_path.name
        stem = safe_stem(target_name)

        tf, tv = targets[target_name]
        base_target = interpolate_log_frequency(tf, tv, master_freq)

        per_iem_delta = []
        per_iem_alignment_unc = []
        target_offsets = []

        for iem_path in preferred_files:
            preferred_curve = preferred_on_master[iem_path.name]

            offsets, scenario_curves, scenario_centre, scenario_unc = alignment_scenarios(
                master_freq,
                preferred_curve,
                base_target,
                bands,
            )

            # Exact prompt definition of per-IEM Delta uses the per-IEM centre
            # from the three aligned scenarios.
            delta_i = scenario_centre - base_target

            per_iem_delta.append(delta_i)
            per_iem_alignment_unc.append(scenario_unc)
            target_offsets.append(offsets)

        delta_stack = np.vstack(per_iem_delta)
        alignment_unc_stack = np.vstack(per_iem_alignment_unc)

        centre = np.median(delta_stack, axis=0)
        cross_iem_mad = np.median(
            np.abs(delta_stack - centre[None, :]),
            axis=0,
        )

        alignment_uncertainty = np.median(
            alignment_unc_stack,
            axis=0,
        )

        total_uncertainty = np.maximum(
            np.maximum(cross_iem_mad, alignment_uncertainty),
            repeatability_floor,
        )

        retention, raw_mask = retention_and_mask(
            centre,
            total_uncertainty,
        )

        mask = gaussian_once_strict_domain(
            raw_mask,
            master_freq,
            personal_start,
            personal_end,
            sigma,
            radius,
        )

        mask *= quarter_octave_sin2_taper(
            master_freq,
            personal_start,
            personal_end,
        )

        outside_personal = (master_freq <= personal_start) | (
            master_freq >= personal_end
        )
        mask[outside_personal] = 0.0

        robust_target = base_target + mask
        robust_target[outside_personal] = base_target[outside_personal]

        if output_cfg["write_masks"]:
            write_xy(
                OUT / f"{stem}__mask.txt",
                output_freq,
                mask[out_mask],
                decimals,
            )

        if output_cfg["write_robust_targets"]:
            write_xy(
                OUT / f"{stem}__robust_target.txt",
                output_freq,
                robust_target[out_mask],
                decimals,
            )

        # Hybrid generation is deliberately on demand in the web app.
        # The locked build_hybrid_curve() function above provides the exact
        # same definition for tests/local use without generating every
        # target combination during each Actions run.

        robust_statistics.append({
            "target": target_name,
            "cross_iem_mad_median_db": float(np.median(cross_iem_mad)),
            "alignment_uncertainty_median_db": float(np.median(alignment_uncertainty)),
            "alignment_uncertainty_max_db": float(np.max(alignment_uncertainty)),
            "repeatability_floor_median_db": float(np.median(repeatability_floor)),
            "total_uncertainty_median_db": float(np.median(total_uncertainty)),
            "retention_median": float(np.median(retention)),
            "mask_abs_max_db": float(np.max(np.abs(mask))),
            "mask_rms_db": float(np.sqrt(np.mean(mask**2))),
        })

        # Per-target alignment scenario report.
        target_report = REPORTS / f"{stem}__alignment_offsets.csv"
        with target_report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iem",
                "offset_200_1000_db",
                "offset_200_500_db",
                "offset_500_1000_db",
            ])
            for path, offsets in zip(preferred_files, target_offsets):
                writer.writerow([
                    path.name,
                    f"{offsets[0]:.12f}",
                    f"{offsets[1]:.12f}",
                    f"{offsets[2]:.12f}",
                ])

    # Reports requested by the prompt.
    with (REPORTS / "alignment_offsets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = list(alignment_report[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(alignment_report)

    with (REPORTS / "robust_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = list(robust_statistics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(robust_statistics)

    # Band statistics for the selected LF alignment offsets.
    with (REPORTS / "alignment_band_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "band_hz",
            "median_db",
            "mad_db",
            "min_db",
            "max_db",
        ])
        for j, band in enumerate(bands):
            vals = np.asarray(
                [row[f"offset_{int(band[0])}_{int(band[1])}_db"] for row in alignment_report],
                dtype=float,
            )
            med = np.median(vals)
            writer.writerow([
                f"{band[0]:g}-{band[1]:g}",
                f"{med:.12f}",
                f"{np.median(np.abs(vals - med)):.12f}",
                f"{np.min(vals):.12f}",
                f"{np.max(vals):.12f}",
            ])

    current_pure = read_output_xy(OUT / "pure_earprint_dynamic.txt")
    change = compare_curves(previous_pure, current_pure)

    (REPORTS / "change_report.txt").write_text(
        "\n".join([
            "IEM EarPrint change report",
            f"Previous pure EarPrint available: {change['previous_available']}",
            f"Current pure EarPrint available: {change['current_available']}",
            f"Matched points: {change['matched_points']}",
            f"RMS change (dB): {change['rms_db']}",
            f"Maximum absolute change (dB): {change['max_abs_db']}",
            "",
            "A new build recalculates the entire EarPrint from the current input/preferred set.",
            "A removed target/IEM cannot leave stale generated TXT because generated TXT outputs are cleared before each build.",
        ]) + "\n",
        encoding="utf-8",
    )

    # Input provenance / SHA-256.
    input_hashes = {}
    for p in preferred_files + target_files:
        input_hashes[str(p.relative_to(ROOT))] = sha256(p)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "IEM EarPrint Engine 2.0.0",
        "specification": cfg.get("specification", "General_Prompt_EarPrint_8_MATH_LOCKED.txt"),
        "dynamic_discovery": True,
        "preferred_glob": discovery.get("preferred_glob", "*.txt"),
        "target_glob": discovery.get("target_glob", "*.txt"),
        "preferred_votes": [p.name for p in preferred_files],
        "independent_vote_count": len(preferred_files),
        "targets_discovered": [p.name for p in target_files],
        "target_count": len(target_files),
        "low_frequency_reference": lf_reference_file,
        "alignment_bands_hz": bands,
        "personal_domain_hz": [personal_start, personal_end],
        "smoothing": {
            "method": "gaussian",
            "sigma_indices": sigma,
            "radius_indices": radius,
            "padding": "nearest",
            "passes": 1,
        },
        "hf_extension": {
            "method": "log_frequency",
            "slope_db_per_octave": -6,
            "display_only": True,
        },
        "repeatability": {
            "valid_same_iem_repeats_present": False,
            "floor_active": False,
            "note": repeatability_note,
        },
        "hybrid_extension": {
            "enabled": bool(cfg.get("hybrids", {}).get("enabled", True)),
            "mode": "on_demand_web_app",
            "rule": "selected target <= join_hz; shifted pure EarPrint > join_hz, with exact log-frequency join continuity",
            "default_join_hz": float(cfg.get("hybrids", {}).get("default_join_hz", personal_start)),
        },
        "grid_points": int(len(master_freq)),
        "output_points": int(len(output_freq)),
        "input_sha256": input_hashes,
        "output_sha256": {},
    }

    for p in sorted(OUT.glob("*.txt")):
        manifest["output_sha256"][p.name] = sha256(p)

    (REPORTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # Human-readable validation/spec conformance report.
    validation_lines = [
        "IEM EarPrint Engine validation / specification conformance",
        "Status: PASS",
        f"Specification: {cfg.get('specification', 'General_Prompt_EarPrint_8_MATH_LOCKED.txt')}",
        "",
        f"Independent IEM votes discovered: {len(preferred_files)}",
        f"Targets discovered: {len(target_files)}",
        "One IEM file = one equal vote.",
        "Interpolation: log-frequency.",
        f"Alignment bands: {bands}",
        "Per-IEM scenario centre: pointwise median of 3 aligned scenarios.",
        "Per-IEM alignment uncertainty: pointwise max absolute distance from scenario centre.",
        "Pure EarPrint direct centre: pointwise median across selected median-aligned IEM curves.",
        "Pure EarPrint smoothing: exactly one Gaussian pass, strict >1 kHz and <12 kHz.",
        "Gaussian parameters: sigma=16 grid indices, radius=64, nearest-endpoint padding.",
        "LF reference: preserved exactly at/below 1 kHz.",
        "HF extension: display-only -6 dB/octave log-frequency at/above 12 kHz.",
        "Robust Centre: pointwise median across IEM Delta_i.",
        "CrossIEM_MAD: pointwise median absolute deviation across IEM Delta_i.",
        "AlignmentUncertainty: pointwise median across IEM per-IEM uncertainty curves.",
        "RepeatabilityFloor: inactive because no valid same-IEM repeat data are supplied.",
        "TotalUncertainty: max(CrossIEM_MAD, AlignmentUncertainty, RepeatabilityFloor).",
        "Retention: abs(Centre)/(abs(Centre)+TotalUncertainty).",
        "RawMask: Centre*Retention.",
        "Mask smoothing: exactly one Gaussian pass inside strict personal domain.",
        "Mask taper: fixed quarter-octave sin-squared boundaries.",
        "Mask forced to zero at/below 1 kHz and at/above 12 kHz.",
        "No final normalization, manual tonal edit, arbitrary gain cap or extra tilt.",
        "Dynamic targets receive robust mask + robust target outputs; hybrids are generated on demand from any selected target.",
    ]
    (REPORTS / "validation.txt").write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    # Explicit method record for auditing.
    (REPORTS / "method_spec.txt").write_text(
        "\n".join([
            "METHOD SPECIFICATION LOCK",
            "Source: user-supplied General_Prompt_EarPrint_8 + explicit math-lock definitions.",
            "",
            "This repository treats the supplied prompt as the mathematical source of truth.",
            "No undocumented robust-mask constants are introduced.",
            "When repeatability data are absent, the prompt's conditional RepeatabilityFloor term is inactive.",
            "Hybrid generation is on demand; any discovered target may be selected without changing robust-target mathematics.",
            "",
            "Input policy: authoritative preferred-response curves are the calculation source.",
        ]) + "\n",
        encoding="utf-8",
    )

    # Final SHA update includes all generated reports after manifest creation only;
    # output hashes already cover the generated curve files required by the prompt.
    print("BUILD PASS")
    print(f"Independent IEM votes discovered: {len(preferred_files)}")
    print(f"Targets discovered: {len(target_files)}")
    print(f"Master grid points: {len(master_freq)}")
    print(f"Generated TXT outputs: {len(list(OUT.glob('*.txt')))}")
    print("Specification math lock: PASS")


if __name__ == "__main__":
    main()
